import os
import utils
import torch
import torch.nn as nn
import lightning.pytorch as L
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from datasets import ProteinDataModule

os.chdir(os.path.expanduser("~"))
torch.set_float32_matmul_precision('high')

class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_raw = acts.clone()
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        if self.cfg.get("raw", False):
            output["acts_raw"] = acts_raw
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)



class LitBatchTopKSAE(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)  
        self.cfg = cfg
        self.sae = BatchTopKSAE(cfg)
        self.test_step_outputs = []

    def forward(self, data):
        x = data.x[:, :self.cfg["act_size"]]
        return self.sae(x)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output["loss"]

        self.log_dict({
            "train_loss": loss,
            "train_l1": output["l1_loss"],
            "train_l2": output["l2_loss"],
            "train_l0": output["l0_norm"],
            "train_l1_norm": output["l1_norm"],
        }, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = output["loss"]

        self.log_dict({
            "val_loss": loss,
            "val_l1": output["l1_loss"],
            "val_l2": output["l2_loss"],
            "val_l0": output["l0_norm"],
            "val_l1_norm": output["l1_norm"],
        }, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))

        return loss
    
    def test_step(self, batch, batch_idx):
        output = self(batch)

        if self.cfg.get("raw", False):
            dasf_ids = self.cfg.get("dasf_ids", None)
            if self.cfg.get("feat_ids", None) is not None:
                acts_raw = output["acts_raw"][:, self.cfg["feat_ids"]]
                self.test_step_outputs.append({
                    "prot_idx": batch.prot_idx,  # [batch_size]
                    "seq_len": torch.bincount(batch.batch).cpu(),  # [batch_size]
                    "acts_topk": acts_raw # [num_nodes, num_features]
                })
                return output["loss"]
            
            elif dasf_ids is not None:
                acts_raw = output["acts_raw"]
                dasf_feats = []
                for domain, indices in dasf_ids.items():
                    selected = acts_raw[:, indices]  # [num_nodes, topk]
                    domain_sum = selected.sum(dim=1, keepdim=True)  # [num_nodes, 1]
                    dasf_feats.append(domain_sum)
                dasf_feats = torch.cat(dasf_feats, dim=1)

                self.test_step_outputs.append({
                    "prot_idx": batch.prot_idx,
                    "seq_len": torch.bincount(batch.batch).cpu(),
                    "acts_topk": dasf_feats  # [num_nodes, num_domains]
                })

                return output["loss"]
            
            else:
                acts_raw = output["acts_raw"].detach().cpu()  # [num_nodes, dict_size]
                nonzero_idx = (acts_raw.abs() > 1e-5).nonzero(as_tuple=False)  # [num_nonzero, 2]
                nonzero_val = acts_raw[nonzero_idx[:, 0], nonzero_idx[:, 1]]  # [num_nonzero]

        else:
            acts_topk = output["feature_acts"].detach().cpu()  # [num_nodes, dict_size]
            nonzero_idx = (acts_topk > 0).nonzero(as_tuple=False)  # [num_nonzero, 2]
            nonzero_val = acts_topk[nonzero_idx[:, 0], nonzero_idx[:, 1]]  # [num_nonzero]

        self.test_step_outputs.append({
            "prot_idx": batch.prot_idx,   # [batch_size]
            "seq_len": torch.bincount(batch.batch).cpu(),   # [batch_size]
            "nonzero_idx": nonzero_idx,  # [nnz, 2]
            "nonzero_val": nonzero_val  # [nnz]
        })
        return output["loss"]
    

    def on_test_epoch_end(self):

        os.makedirs("code/sae", exist_ok=True)

        if any(self.cfg.get(k) is not None for k in ("feat_ids", "dasf_ids")):
            all_prot_idx = []
            all_seq_len = []
            all_acts_topk = []

            for output in self.test_step_outputs:
                all_prot_idx.extend(output["prot_idx"])
                all_seq_len.extend(output["seq_len"].tolist())
                all_acts_topk.append(output["acts_topk"])

            prot_lens = dict(zip(all_prot_idx, map(int, all_seq_len)))

            all_acts_topk = torch.cat(all_acts_topk, dim=0)  # [total_num_nodes, num_features]

            torch.save({
                "prot_lens": prot_lens,
                "acts_topk": all_acts_topk
            }, self.cfg.get("save_path", "code/sae/activations.pt"))

        else:
            # Initialize the list for concatenation
            all_prot_idx = []
            all_seq_len = []
            all_nonzero_idx = []
            all_nonzero_val = []

            offset = 0
            # Concatenate results from all batches
            for output in self.test_step_outputs:
                seq_lens = output["seq_len"]
                all_seq_len.extend(seq_lens.tolist())
                all_prot_idx.extend(output["prot_idx"])

                idx = output["nonzero_idx"]
                idx[:, 0] += offset
                all_nonzero_idx.append(idx)
                all_nonzero_val.append(output["nonzero_val"])

                offset += seq_lens.sum().item()

            all_nonzero_idx = torch.cat(all_nonzero_idx, dim=0)
            all_nonzero_val = torch.cat(all_nonzero_val, dim=0)

            num_nodes = offset
            num_features = all_nonzero_idx[:, 1].max().item() + 1

            sparse_tensor = torch.sparse_coo_tensor(
                indices=all_nonzero_idx.t(),  # shape: [2, N]
                values=all_nonzero_val,
                size=(num_nodes, num_features)
            )

            prot_lens = dict(zip(all_prot_idx, map(int, all_seq_len)))

            if self.cfg.get("raw", False):
                file_name = "code/sae/test_activations_raw.pt"
            else:
                file_name = "code/sae/test_activations_sparse.pt"

            offsets = {}
            offset = 0
            for prot_id, length in prot_lens.items():
                offsets[prot_id] = (offset, offset + length)
                offset += length

            torch.save({
                "prot_lens": prot_lens,
                "sparse_acts": sparse_tensor,
                "offsets": offsets
            }, file_name)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=self.cfg["lr"],
            betas=(self.cfg["beta1"], self.cfg["beta2"]),
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Project gradients and weights to unit norm before the optimizer step
        self.sae.make_decoder_weights_and_grad_unit_norm()



def get_default_cfg():
    default_cfg = {
        "sae_type": "batchtopk",
        "seed": 49,
        "lr": 3e-4,
        "l1_coeff": 0.004,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "act_size": 1280,
        "dict_size": 1280*8,
        "device": "cuda:0",
        "input_unit_norm": True,

        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,

        # (Batch)TopKSAE specific
        "top_k": 128,
        "top_k_aux": 1024,
        "aux_penalty": (1/128),

        "feat_ids": None,
    }

    return default_cfg


if __name__ == "__main__":
    cfg = get_default_cfg()
    device = utils.get_available_gpu()
    cfg["device"] = device
    accelerator = 'cpu' if str(device) == 'cpu' else 'gpu'

    tb_logger = TensorBoardLogger("code/tb_logs", name='sae', version=0)

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='code/model/sae',
        filename="sae-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=2, 
        verbose=True
    )

    L.seed_everything(cfg["seed"])


    datamodeule = ProteinDataModule(task_type='pretrain',
                                    y_type = 'str',
                                    concat = True,
                                    balance = 'no', 
                                    batch_size=64, num_workers=0,
                                    update_label=True,
                                    model_name = 'esm2_t33_650M_UR50D',
                                    layer_num = 33,
                                    hub_dir = 'code/esm2_t33_650M_UR50D',
                                    )
    datamodeule.setup('train_test')
    # train_loader = datamodeule.train_dataloader()  # train 
    # train_loader = datamodeule.train_dataloader(shuffle=False)# CFS
    train_loader =  datamodeule.all_dataloader() # caculate all activations



    trainer = L.Trainer(
        max_epochs=3,
        callbacks=[checkpoint_callback, early_stopping_callback], 
        accelerator=accelerator,
        logger=tb_logger,
        gradient_clip_val=cfg["max_grad_norm"],
        log_every_n_steps=100,
    )

    # model = LitBatchTopKSAE(cfg)
    # trainer.fit(model, train_loader)

    cfg["raw"] = True  # Whether to use raw activation values

    cfg["dasf_ids"] = {
        "KH":    [9079, 6861, 6895, 5300, 9203],
        "RRM":   [3017, 7315, 9203, 2575, 3673],
        "DEAD":  [8999, 131, 9240, 2540, 8890],
        "CSD":   [6915, 9707, 3806, 2506, 9991],
        "LSM":   [779, 4364, 6326, 4233, 7916]
    }  


    model = LitBatchTopKSAE.load_from_checkpoint("code/model/sae/sae-epoch=02-train_loss=0.40.ckpt", 
                                                 cfg=cfg, strict=False
                                                 )
    trainer.test(model, train_loader)



