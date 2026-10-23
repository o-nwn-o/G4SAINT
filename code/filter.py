import os
import re
import torch
import utils
import numpy as np
import pandas as pd
import lightning.pytorch as L
from torch_geometric.loader import DataLoader
from torch_geometric.data.dataset import Dataset
from torch_geometric.nn import global_add_pool, global_mean_pool

from model import G4SAINT
from pdb2graph import PDB2Graph
from SAE import LitBatchTopKSAE, get_default_cfg

def collate_with_interpro(inter_features):
    indices, indptr, data = inter_features
    pairs = indptr.reshape(-1, 2)  # shape [n, 2]
    offsets = [pairs[0, 0].item()]
    for i in range(pairs.shape[0]):
        offsets.append(offsets[-1] + pairs[i, 1].item())
    offsets = torch.tensor(offsets, dtype=torch.long, device=indices.device)
    return (indices, offsets, data)


class DynamicDomainEncoder:
    def __init__(self, interpro_tsv, evalue_threshold=1e-1):
        domain_tsv = "data/domain.tsv"
        self.domain_df = pd.read_csv(domain_tsv, sep='\t', dtype=str)
        self.domain_df = self.domain_df[~self.domain_df['Integrated Into'].isna()]
        self.domain_order = sorted(self.domain_df['Integrated Into'].unique().tolist())
        self.ipr_to_index = {ipr: i for i, ipr in enumerate(self.domain_order)}
        self.n_domains = len(self.domain_order)

        interpro_cols = ['Sequence ID', 'MD5 Checksum', 'Sequence Length', 'Analysis Method',
                        'Accession Number', 'Domain Name', 'Start Position', 'End Position',
                        'E-value', 'Status', 'Date', 'InterPro Entry ID', 'InterPro Entry Description',
                        'GO Terms', 'Pathway Entry']
        
        self.interpro_df = pd.read_csv(interpro_tsv, sep='\t', names=interpro_cols, dtype=str)
        self.interpro_df['Uniprot ID'] = self.interpro_df['Sequence ID'].apply(
            lambda x: x.split('|')[1] if '|' in x else x
        )
        self.interpro_df.dropna(subset=['InterPro Entry ID'], inplace=True)

        if evalue_threshold is not None:
            self.interpro_df['E-value'] = pd.to_numeric(self.interpro_df['E-value'], errors='coerce')
            self.interpro_df = self.interpro_df[self.interpro_df['E-value'] < evalue_threshold]

    def get_sparse_csr_for_uniprot(self, uniprot_id):
    
        df = self.interpro_df[self.interpro_df['Uniprot ID'] == uniprot_id]
        iprs = [ipr for ipr in df['InterPro Entry ID'] if ipr in self.ipr_to_index]
        col_indices = sorted(set(self.ipr_to_index[ipr] for ipr in iprs))

        indices = torch.tensor(col_indices, dtype=torch.long)
        indptr = torch.tensor([0, len(indices)], dtype=torch.long)
        data = torch.ones(len(indices), dtype=torch.float)
        return indices, indptr, data


class DynamicProteinGraphDataset(Dataset):
    def __init__(self, pdb_paths, **kwargs):
        self.pdb_paths = pdb_paths
        self.graph_builder = PDB2Graph(
            model_name = 'esm2_t33_650M_UR50D',
            layer_num = 33,
            hub_dir = 'code/esm2_t33_650M_UR50D',
        )
        self.device = self.graph_builder.device
        self.interpro_builder = None
        self.sae_model = None
        self.common_params = {
            'ESM': True, 'loop': True,
            'k': 10, 'd_cutoff': 8.0, **kwargs
        }
        self.kwargs = kwargs

        if kwargs.get("use_domain", False):
            interpro_output_file = kwargs.get("interpro_output_file")
            self.interpro_builder = DynamicDomainEncoder(interpro_output_file)
        cfg = get_default_cfg()
        cfg.update({
            "device": self.device,
            "dasf_ids": kwargs.get("dasf_ids"),
            "raw": True,
        })
        self.cfg = cfg
        sae_checkpoint = kwargs.get("sae_checkpoint")
        self.sae_model = LitBatchTopKSAE.load_from_checkpoint(
            sae_checkpoint,
            cfg=self.cfg, strict=False
        )
        self.sae_model.to(self.device)

    def __len__(self):
        return len(self.pdb_paths)

    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        try:
            structure_id = re.search(r'AF-(.*?)-F1', pdb_path).group(1)
            prot_idx = structure_id
            data = self.graph_builder(pdb_path, **self.common_params)
            data.to(self.device)
            if self.interpro_builder:
                indices, indptr, values = self.interpro_builder.get_sparse_csr_for_uniprot(prot_idx)
                data.inter_features = (indices, indptr, values)
            if self.sae_model:
                with torch.no_grad():
                    output = self.sae_model(data)
                    acts_raw = output["acts_raw"]
                    dasf_feats = []
                    for domain, indices in self.cfg["dasf_ids"].items():
                        selected = acts_raw[:, indices]  # [num_nodes, topk]
                        domain_sum = selected.sum(dim=1, keepdim=True)  # [num_nodes, 1]
                        dasf_feats.append(domain_sum)
                    dasf_feats = torch.cat(dasf_feats, dim=1)
                    data.acts_topk = dasf_feats
            return data
        except Exception as e:
            print(f"Error parsing {pdb_path}: {e}")
            raise IndexError
    

class G4SAINTFilter(G4SAINT):
    def __init__(self, params):
        super().__init__(params)

    def forward(self, data, class_name=None):
        edge_index, batch = data.edge_index, data.batch
        acts_topk = getattr(data, 'acts_topk', None)
        inter_features = collate_with_interpro(data.inter_features)
        x_pool = global_mean_pool(data.x, batch)

        inter_features = self.embedding_layer(*inter_features)
        readout, ana = self.gnn(data.x, edge_index, inter_features, batch, acts_topk)
        readout = global_add_pool(readout, batch)
        readout = torch.cat([readout, x_pool], dim=-1)
        logits_dict = self.fc(readout)
        logits = torch.cat([logits_dict[task] for task in self.active_task_names], dim=1)
        # Retrieve output for a single target category
        if class_name is not None:
            if class_name not in self.active_task_names:
                raise ValueError(f"{class_name} not in active_task_names")
            class_index = self.active_task_names.index(class_name)
            logits = logits[:, class_index]
        return logits
    
    def test_step(self, data, batch_idx):
        with torch.no_grad():
            logits = self(data)
            self.test_step_outputs.append({
            'logits': logits.detach(),
            'id': data.prot_idx
        })
    
    def on_test_epoch_end(self):
        logits = torch.cat([x['logits'] for x in self.test_step_outputs], dim=0)  # 合并所有 logits
        ids = sum([x['id'] for x in self.test_step_outputs], [])
        self.test_step_outputs.clear()
        logits = logits.cpu().detach().numpy()
        if logits.ndim == 1:  # Single-class prediction results
            df = pd.DataFrame({
                'id': ids,
                'logit': logits
            })
        else:  # Multi-class prediction results
            df = pd.DataFrame({
                'id': ids,
                **{f'logit_{task}': logits[:, i] for i, task in enumerate(self.active_task_names)},
            })
        # df.to_csv("code/filter_outputs.csv", index=False)


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~"))
    device = utils.get_available_gpu()
    accelerator = 'cpu' if str(device) == 'cpu' else 'gpu'
    torch.set_float32_matmul_precision('high')
    trainer = L.Trainer(max_epochs=10, accelerator=accelerator,)
    best_model_path = "code/model/g4saint/g4saint_predictor.ckpt"
    test_model = G4SAINTFilter.load_from_checkpoint(best_model_path, strict=False)
    test_model.to(device)

    data_params = {
        "interpro_output_file": "data/TrEMBL/interproscan_pfam_output.tsv",
        "sae_checkpoint": "code/model/sae/g4saint_sae.ckpt",
        "dasf_ids": {
            "KH":    [9079, 6861, 6895, 5300, 9203],
            "RRM":   [3017, 7315, 9203, 2575, 3673],
            "DEAD":  [8999, 131, 9240, 2540, 8890],
            "CSD":   [6915, 9707, 3806, 2506, 9991],
            "LSM":   [779, 4364, 6326, 4233, 7916]
        },
        "use_domain": False,
        "use_sae": False,
    }

    data_params['use_domain']=test_model.params["use_domain"]
    data_params['use_sae']=test_model.params["use_sae"]
    folder = "data/TrEMBL"
    # pdb_names = ["AF-B2R8Z8-F1-model_v4.pdb", "AF-Q5T6W2-F1-model_v4.pdb"]
    # pdb_files = [os.path.join(folder, name) for name in pdb_names]

    pdb_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdb")]
    dataset = DynamicProteinGraphDataset(pdb_files, **data_params)
    dataloader = DataLoader(dataset, batch_size=1)

    trainer.test(test_model, dataloader)