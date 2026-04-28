import os
import torch
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

os.chdir(os.path.expanduser("~"))

class InterProData:
    def __init__(self, uniprot_ids_file=None, interpro_output_file=None, save_dir=None):
        domain_file="data/domain.tsv"
        self.uniprot_to_idx = {}
        self.idx_to_uniprot = {}
        self.ipr_to_index = {}
        self.idx_to_ipr = {}
        self.interpro_matrix = None

        # Automatically generate save_dir
        if not save_dir and interpro_output_file:
            self.save_dir = os.path.join(os.path.dirname(interpro_output_file), "interpro_data")
        else:
            self.save_dir = save_dir

        # Automatically detect and load/generate data
        if self.data_exists():
            self.load_data()
        elif uniprot_ids_file and domain_file and interpro_output_file:
            self.generate_and_save_data(uniprot_ids_file, domain_file, interpro_output_file)
        else:
            raise ValueError("No saved data found and missing input files required to generate data.")

    @property
    def processed_dir(self) -> str:
        """Return the path where processed data is saved"""
        return self.save_dir

    @property
    def processed_file_names(self):
        """Return the names of the saved files"""
        return {
            'matrix': 'interpro_matrix.pkl',
            'uniprot_idx': 'uniprot_to_idx.pkl',
            'ipr_idx': 'ipr_to_index.pkl'
        }

    def data_exists(self) -> bool:
        """Check if data files exist in save_dir"""
        return all(
            Path(self.processed_dir, filename).exists() 
            for filename in self.processed_file_names.values()
        )

    def generate_and_save_data(self, uniprot_ids_file, domain_file, 
                               interpro_output_file, evalue_threshold=1e-1):
        # Step 1: Read the Uniprot ID list
        with open(uniprot_ids_file, 'r') as f:
            uniprot_ids = [line.strip() for line in f.readlines()]
        
        self.uniprot_to_idx = {uniprot: idx for idx, uniprot in enumerate(uniprot_ids)}
        self.idx_to_uniprot = {idx: uniprot for uniprot, idx in self.uniprot_to_idx.items()}

        # Step 2: Load the complete InterPro domain index
        domain_df = pd.read_csv(domain_file, sep='\t', dtype=str)
        domain_df = domain_df[~domain_df['Integrated Into'].isna()]
        all_ipr_ids = sorted(domain_df['Integrated Into'].unique().tolist())

        self.ipr_to_index = {ipr: idx for idx, ipr in enumerate(all_ipr_ids)}
        self.idx_to_ipr = {idx: ipr for ipr, idx in self.ipr_to_index.items()}

        # Step 3: Read the InterProScan output file
        columns = ['Sequence ID', 'MD5 Checksum', 'Sequence Length', 'Analysis Method',
                   'Accession Number', 'Domain Name', 'Start Position', 'End Position',
                   'E-value', 'Status', 'Date', 'InterPro Entry ID', 'InterPro Entry Description',
                   'GO Terms', 'Pathway Entry']

        scan_df = pd.read_csv(interpro_output_file, sep='\t', names=columns, dtype=str)
        scan_df['Uniprot ID'] = scan_df['Sequence ID'].apply(lambda x: x.split('|')[1] if '|' in x else x)

        if evalue_threshold is not None:
            scan_df['E-value'] = pd.to_numeric(scan_df['E-value'], errors='coerce')
            scan_df = scan_df[scan_df['E-value'] < evalue_threshold]
                
        # Step 4: Create a sparse matrix
        rows, cols, data = [], [], []

        for _, row in tqdm(scan_df.iterrows(), total=scan_df.shape[0]):
            uniprot_id = row['Uniprot ID']
            accession = row['Accession Number']

            ipr_id = domain_df.loc[domain_df['Accession'] == accession, 'Integrated Into'].values
            if len(ipr_id) == 0:
                continue  
            ipr_id = ipr_id[0]

            if uniprot_id in self.uniprot_to_idx and ipr_id in self.ipr_to_index:
                rows.append(self.uniprot_to_idx[uniprot_id])
                cols.append(self.ipr_to_index[ipr_id])
                data.append(1)

        self.interpro_matrix = csr_matrix((data, (rows, cols)),
                                          shape=(len(uniprot_ids), len(all_ipr_ids)))

        # Step 5: Save data
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        for name, obj in zip(
            ['matrix', 'uniprot_idx', 'ipr_idx'],
            [self.interpro_matrix, self.uniprot_to_idx, self.ipr_to_index]
        ):
            with open(Path(self.processed_dir) / self.processed_file_names[name], 'wb') as f:
                pkl.dump(obj, f)

        print(f"Data saved to {self.processed_dir}")

    def load_data(self):
        """Load data from save_dir"""
        with open(Path(self.processed_dir) / self.processed_file_names['matrix'], 'rb') as f:
            self.interpro_matrix = pkl.load(f)
        with open(Path(self.processed_dir) / self.processed_file_names['uniprot_idx'], 'rb') as f:
            self.uniprot_to_idx = pkl.load(f)
        with open(Path(self.processed_dir) / self.processed_file_names['ipr_idx'], 'rb') as f:
            self.ipr_to_index = pkl.load(f)

        self.idx_to_uniprot = {idx: uniprot for uniprot, idx in self.uniprot_to_idx.items()}
        self.idx_to_ipr = {idx: ipr for ipr, idx in self.ipr_to_index.items()}

        print(f"Data loaded from {self.processed_dir}")

    def get_interpro_annotations_by_uniprot(self, uniprot_id):
        """Extract InterPro annotations for the specified Uniprot ID from the interpro_matrix"""
        if uniprot_id not in self.uniprot_to_idx:
            raise ValueError(f"Uniprot ID {uniprot_id} does not exist in the sparse matrix.")

        row_idx = self.uniprot_to_idx[uniprot_id]
        features = self.interpro_matrix[row_idx].toarray().flatten()
        annotated_iprs = [self.idx_to_ipr[idx] for idx in np.nonzero(features)[0]]
        return annotated_iprs
    
    def get_interpro_features_by_indices(self, sample_idx):
        """
        Extract InterPro feature information based on sample indices (sample_idx).
        Returns (indices, indptr, data) in a format compatible with PyTorch sparse matrices.
        """
        # Extract the sparse matrix fragment for the specified indices
        sub_matrix = self.interpro_matrix[sample_idx]

        # Extract the indices, indptr, and data of the sparse matrix fragment
        indices = torch.tensor(sub_matrix.indices, dtype=torch.long)
        indptr = torch.tensor(sub_matrix.indptr, dtype=torch.long)
        data = torch.tensor(sub_matrix.data, dtype=torch.float)
        return (indices, indptr, data)
    
    def get_csr_submatrix(self, sample_idx):
        """
        Return a subset of the InterPro sparse matrix (scipy.sparse.csr_matrix) for the specified sample indices.
        """
        return self.interpro_matrix[sample_idx]

    

if __name__ == "__main__":
    data_manager = InterProData(
        uniprot_ids_file="data/uniprotkb_ids.txt",
        interpro_output_file="data/uniprotkb/interproscan_pfam_output.tsv",
    )

    # # regenerate data if needed
    # data_manager.generate_and_save_data(
    #     "/mnt/disk1/nwn/data/uniprotkb_ids.txt",
    #     "data/domain.tsv",
    #     "/mnt/disk1/nwn/data/uniprotkb/interproscan_pfam_output.tsv",
    #     )

    # # Query InterPro features for a Uniprot ID
    uniprot_id = ['P41091', 'Q9UGR2']
    sample_idx = [data_manager.uniprot_to_idx[pid] for pid in uniprot_id]
    inter_features = data_manager.get_interpro_features_by_indices(sample_idx)
    for feat in inter_features:
        print(feat)
        print(len(feat))
    # features = data_manager.get_interpro_features_by_indices(sample_idx)
    # print(f"InterPro annotation features for {uniprot_id}: {features}")