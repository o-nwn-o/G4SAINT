import os
import gc
import re
import csv
import json
import copy
import glob
import utils
import torch
import pickle
import shutil
import random
import numpy as np
import pandas as pd
import seaborn as sns
import lightning.pytorch as L
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from collections import Counter
from sklearn.utils import resample
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.data.separate import separate
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch.utils.data import Dataset, ConcatDataset, Subset
from transformers import AutoTokenizer, EsmModel

from pdb2graph import PDB2Graph
from data_division import ClusterDataManager

os.chdir(os.path.expanduser("~"))


def calculate_average_embeddings(dataset):
    # Create a DataLoader to iterate through the dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    average_embeddings = []
    for batch in loader:
        # Assume that each batch contains data where 'embeddings' holds the embedding data
        embeddings = batch.squeeze(0)
        average_embedding = embeddings.mean(dim=0)
        average_embeddings.append(average_embedding)
    return average_embeddings

# Example usage
# Assume there is a dataset instance named dataset
# average_embeddings = calculate_average_embeddings(dataset)
    

def calculate_label_proportion(dataset, lower_bound, upper_bound):
    count = 0
    total = 0

    # If the dataset is a ConcatDataset, iterate through its subdatasets
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            for data in sub_dataset:
                label = data.y.item()
                if lower_bound < label < upper_bound:
                    count += 1
                total += 1
    else:
        for data in dataset:
            label = data.y.item()
            if lower_bound < label < upper_bound:
                count += 1
            total += 1

    proportion = count / total if total > 0 else 0
    return proportion, count, total


def get_attr(data_list, attr_name: str):
    """
    Merge a specified graph-level attribute from all graphs in the dataset (only merge the same attribute from different graphs).

    Args:
        data_list (list): List of datasets.
        attr_name (str): The name of the graph-level attribute to merge.

    Returns:
        torch.Tensor or list: If the attribute is numeric, returns a tensor of shape [num_graphs, feature_dim]; if the attribute is a string, returns a list of strings.
    """

    allowed_attributes = ['y', 'molecule_features', 'prot_idx', 'sequence']

    if attr_name not in allowed_attributes:
        raise ValueError(f"'{attr_name}' is not in the allowed attributes list {allowed_attributes}")

    merged_attributes = []

    for data in data_list:
        if hasattr(data, attr_name):
            attr_value = getattr(data, attr_name)

            # Handle numeric types (e.g., y, molecule_features)
            if isinstance(attr_value, torch.Tensor):
                attr_value = attr_value.numpy()
            merged_attributes.append(attr_value)
        else:
            print(data)
            raise AttributeError(f"The data object in the dataset lacks the attribute '{attr_name}'")

    # Return Tensor for numeric attributes, List for string attributes
    if isinstance(merged_attributes[0], (np.ndarray, list)):
        merged_attributes = np.array(merged_attributes)
        return torch.tensor(merged_attributes, dtype=torch.float).squeeze(1)
    return merged_attributes


class DatasetWrapper(InMemoryDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.indices = idx  # Save the original index of the sample in data
        return data
    
class DatasetWrapper2(InMemoryDataset):
    def __init__(self, dataset, offsets, all_acts_topk):
        super().__init__()
        self.dataset = copy.deepcopy(dataset)  # Deep copy to avoid modifying the original dataset
        
        self.data_list = []
        for i in range(len(self.dataset)):
            data = self.dataset.get(i)  # Note: using .get(i) here for in-place slicing of small samples
            prot_id = data.prot_idx
            start_idx, end_idx = offsets[prot_id]
            data.acts_topk = all_acts_topk[start_idx:end_idx, :]
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)

# ==============================
# Dataset
# ==============================
class ProteinGraph(InMemoryDataset):
    """
    A PyTorch Geometric dataset for protein graph data.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The transform function can be used for data augmentation or preprocessing.
            (default: None)
        pre_transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The pre_transform function can be used for data augmentation or preprocessing
            that is applied before saving the data to disk. (default: None)
        Hetero (bool, optional): Whether to use the hetero graph for protein graph construction.
            (default: False)
        ESM (bool, optional): Whether to use the ESM embedding for protein graph construction.
            (default: False)
        seq (bool, optional): Whether to include the protein sequence information in the graph.
            (default: False)
        loop (bool, optional): Whether to include loop edges in the graph.
            (default: False)
        k (int, optional): The number of nearest neighbors to consider when constructing the graph.
            (default: -1)
        d_cutoff (float, optional): The distance cutoff for considering two atoms as neighbors.
            (default: 8.0)
        y (int, optional): The target label for the dataset. (default: 1)
        **kwargs: Additional keyword arguments.

    Attributes:
        num (int): The number of datas in the dataset.
        processed_dir (str): The directory where the processed data is saved.
        raw_file_names (str): The root directory of the dataset.
        processed_file_names (list): The list of processed file names.
    """

    def __init__(self, root, pfam_df=None, label_path=None, transform=None, pre_transform=None, 
                 overwrite=False, update_label=False, **kwargs):
        parts = os.path.normpath(root).split(os.sep)
        self.file_names = self.folder = parts[-1] 
        self.raw_file = root
        self.pfam_df = pfam_df
        self.label_path = label_path
        self.overwrite = overwrite
        self.update_label = update_label
        self.kwargs = kwargs
        
        if self.kwargs.get('Hetero', False):
            self.folder += '_h'

        if self.kwargs.get('ESM', False):
            model_name = self.kwargs.get('model_name', 'esm2_t6_8M_UR50D')
            self.folder += f"_{model_name}"

        if self.kwargs.get('add_edge', False):
            self.folder += '_edge'

        if self.kwargs.get('fuz', False):
            self.folder += '_fuz'

        self.prefix = os.path.join(root, self.folder)
        y_value = self.kwargs.get('y', 1)
        if isinstance(y_value, str):
            if self.label_path is None:
                raise ValueError("The value of 'y' as a string requires 'label_path'")
        elif not isinstance(y_value, (int, float, torch.Tensor)):
            raise ValueError("The value of 'y' must be a number, a tensor, or a string when label_path is provided")
        self.folder = self.folder + f"_{y_value}"
        print(f"Dataset folder: {self.folder}")

        if os.path.exists(self.processed_dir) and overwrite: 
            shutil.rmtree(self.processed_dir)

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        if os.path.exists(self.processed_paths[1]):
            with open(self.processed_paths[1], 'r') as f:
                self.structure_ids = [line.strip() for line in f]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.raw_file, self.folder)
    
    @property
    def raw_file_names(self):
        return self.raw_file

    @property
    def processed_file_names(self):
        return [self.file_names+'.pt', self.file_names+'_structure_ids.txt']
    
    def metadata(self):
        node_types = ['protein']
        edge_types = [('protein', 'SEQ', 'protein'), ('protein', 'DIST', 'protein'), ('protein', 'KNN', 'protein')]
        return node_types, edge_types
    
    def process(self):

        src_dir = self.get_valid_folder()

        # ---------- Case 1 ----------
        # overwrite
        if self.overwrite:
            print("Overwrite dataset...")
            self.__process()
            return

        # ---------- Case 2 ----------
        # update label
        if self.update_label:

            if src_dir is not None:
                print("Updating labels from cached dataset...")
                self.update_labels(src_dir, self.processed_dir)
                return

            print("No cached dataset found. Rebuilding...")
            self.__process()
            return

        # ---------- Case 3 ----------
        self.__process()

    def __process(self):
        builder = PDB2Graph(self.pfam_df, self.label_path, **self.kwargs)
        pdb_files = [os.path.join(self.raw_file_names, filename) for filename in os.listdir(self.raw_file_names) if filename.endswith('.pdb')]
        data_list = []
        structure_ids = []

        for idx, file_path in enumerate(tqdm(pdb_files, desc='Processing pdb files')):      

            structure_id = re.search(r'AF-(.*?)-F1', file_path).group(1)
            if structure_id in structure_ids:
                print(f"Warning: Structure ID '{structure_id}' already exists. Skipping file {file_path}.")
                continue
            try:
                data = builder(file_path, **self.kwargs)
                data_list.append(data)
                structure_ids.append(structure_id)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

            # Check if it's time to empty the CUDA cache
            if idx % 500 == 0:
                torch.cuda.empty_cache()


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data = data_list
        self.structure_ids = structure_ids


        torch.save(self.collate(data_list), self.processed_paths[0])
        with open(self.processed_paths[1], 'w') as f:
            for item in structure_ids:
                f.write("%s\n" % item)

    def get_valid_folder(self):
        valid_folders = []
        search_pattern = self.prefix + "*"
        matching_folders = glob.glob(search_pattern)
        required_files = self.processed_file_names
        pattern = re.compile(rf'^{re.escape(self.prefix)}[^_]*_[^_]+$')

        for folder in matching_folders:
            if os.path.isdir(folder) and pattern.match(folder):
                if all(os.path.exists(os.path.join(folder, file)) for file in required_files):
                    valid_folders.append(folder)
        if not valid_folders:
            return None
        latest_folder = max(valid_folders, key=os.path.getmtime)
        return latest_folder
    
    def get_new_y(self):
        if os.path.exists(self.processed_paths[1]):
            with open(self.processed_paths[1], 'r') as f:
                self.structure_ids = [line.strip() for line in f]
        protein_labels = {}
        label_path = os.path.join(os.path.expanduser("~"), self.label_path)
        with open(label_path, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                uniprot_id = row["UniProt_ID"]
                binding_vector = json.loads(row["Binding_Vector_JSON"])
                protein_labels[uniprot_id] = binding_vector

        new_y_list = []
        first_binding_vector = next(iter(protein_labels.values()))
        class_num = len(first_binding_vector)
        for structure_id in self.structure_ids:
            if structure_id in protein_labels:
                binding_vector = protein_labels[structure_id]
            else:
                binding_vector = [0] * class_num
            new_y_list.append(binding_vector)

        return new_y_list


    def update_labels(self, src_dir, dst_dir):
        # Check if src_dir and dst_dir are the same
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
                print("Source and destination directories are the same. Updating in place.")
        else:
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        dataset_path = os.path.join(src_dir, self.processed_file_names[0])
        output_path = self.processed_paths[0]
        self.data, self.slices = torch.load(dataset_path, weights_only=False)
        if os.path.exists(self.processed_paths[1]):
            with open(self.processed_paths[1], 'r') as f:
                self.structure_ids = [line.strip() for line in f]

        y = self.kwargs.get('y', 1)
        new_y = self.kwargs.get('y_tensor', None)
        if isinstance(y, str) and self.label_path is not None:
        # new_y is not None and len(new_y) == len(data.y):
            new_y = self.get_new_y()
            self.data.y = torch.tensor(new_y, dtype=self.data.y.dtype)
        else:
            if not isinstance(y, (int, float, torch.Tensor)):
                raise ValueError("The value of 'y' must be a number or a tensor")
            self.data.y = torch.full((len(self.data.y),), y)


        # Save the updated dataset to the new output path
        torch.save((self.data, self.slices), output_path)
        print(f"Updated dataset saved to {output_path}")


    def __repr__(self) -> str:
        return 'ProteinGraph()'

# Custom Dataset Class
class DualGraphDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = [data.y for data in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UpdateDataset(InMemoryDataset):
    """
    UpdateDataset is a custom dataset class that extends the InMemoryDataset class.
    It is designed to handle the updating and processing of node features in a dataset.
    The class supports loading data from disk, updating node features using specified models,
    and saving the processed data back to disk. It also includes methods for memory management
    and unbatching data.

    Attributes:
        root (str): Root directory for saving and loading data.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        model_RG4BP (torch.nn.Module): Model used for updating node features.
        model_RBP (torch.nn.Module): Another model used for updating node features.
        dataloader (torch.utils.data.DataLoader): DataLoader for loading batches of data.
    """
    def __init__(self, root, device='cpu', model_RG4BP=None, model_RBP=None, dataloader=None, overwrite=False, transform=None, pre_transform=None):
        self.model_RG4BP = model_RG4BP
        self.model_RBP = model_RBP
        self.dataloader = dataloader
        self.device = device
        self.root = root

        if overwrite and os.path.exists(self.root):
            shutil.rmtree(self.root)

        super().__init__(root, transform, pre_transform)
        
        # Load the merged dataset
        self.data, self.slices = torch.load(self.processed_paths[0])
    

    @property
    def raw_file_names(self):
        return []  # No need to define raw data file names

    @property
    def processed_file_names(self):
        return ['updated_data.pt']  # Define the file name for saving processed data
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'updated')


    def process(self):
        if not os.path.exists(self.root):
            self.update_node_features()
        # Check if the save path exists
        batch_files = glob.glob(os.path.join(self.root, "*.pt"))
        if not batch_files:
            # If batch files do not exist, update node features
            print(f"No batch files found in {self.root}, updating node features first.")
            self.update_node_features()  # Update node features first
            batch_files = glob.glob(os.path.join(self.root, "*.pt"))
            if not batch_files:
                raise ValueError(f"Failed to create batch files in {self.root}. Please ensure the update was successful.")

        data_list = self.load_and_unbatch(batch_files)
        
        # Merge all Data objects into a complete dataset
        data, slices = self.collate(data_list)
        
        # Save the merged data to disk
        torch.save((data, slices), self.processed_paths[0])
        print(f"Data has been successfully merged and saved to {self.processed_paths[0]}")


    def update_node_features(self):
        """
        Update node features and save the updated data to disk.
        """

        if self.dataloader is None:
            raise ValueError("Dataloader must be provided for updating node features.")
        
        if self.model_RG4BP is None or self.model_RBP is None:
            raise ValueError("Both model_RG4BP and model_RBP must be provided before initializing UpdateDataset.")


        if os.path.exists(self.root):
            # Find all .pt files
            pt_files = glob.glob(os.path.join(self.root, "*.pt"))

            if pt_files:
                for pt_file in pt_files:
                    try:
                        os.remove(pt_file)
                        print(f"Deleted: {pt_file}")
                    except OSError as e:
                        print(f"Error deleting file {pt_file}: {e}")
            else:
                print(f"No .pt files found in {self.root}.")
        else:
            os.makedirs(self.root)  # Create if it does not exist
        
        for batch_idx, data in enumerate(self.dataloader):
            # Load data to the specified device (GPU/CPU)
            data = data.to(self.device)

            # Extract the last 3 dimensions of node features as extra features
            extra_features = data.x[:, -3:]

            # Update node features through RG4BP and RBP models respectively
            updated_features_RG4BP = self.model_RG4BP.update_node_features(data)
            updated_features_RBP = self.model_RBP.update_node_features(data)

            # Concatenate updated node features with extra features
            updated_features = torch.cat([updated_features_RG4BP, updated_features_RBP, extra_features], dim=-1)

            # Assign the updated node features back to data.x
            data.x = updated_features.cpu()  # Move back to CPU

            # Save to disk
            torch.save(data, f"{self.root}/batch_{batch_idx}.pt")

        # After updating all batches, clean up unused models and datasets, and release memory
        self.clear_memory()


    def clear_memory(self):
        """
        Thoroughly clean up memory, including unused data on GPU and CPU, and Python objects.
        """
        # Delete unused objects
        del self.model_RG4BP
        del self.model_RBP
        del self.dataloader

        # Clean up Python garbage collection
        torch.cuda.empty_cache()  # Clean up unused GPU memory
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        gc.collect()  # Force Python to perform garbage collection
        print("Memory has been successfully cleared.")


    def load_and_unbatch(self, batch_files):
        """
        Load saved batch files from disk and unpack DataBatch into individual Data objects.
        """
        data_list = []
        for batch_file in batch_files:
            # Load saved DataBatch object
            batch_data = torch.load(batch_file, map_location='cpu')

            # Use to_data_list method to unpack batch into individual Data objects
            data_list.extend(Batch.to_data_list(batch_data))

        return data_list



# ==============================
# DataModule
# ==============================
class ProteinDataModule(L.LightningDataModule):
    def __init__(self, task_type='RG4BP_vs_nonNABP', use_esm=True, batch_size=16, 
                 num_workers=0, balance='mix', concat=False, y_type="num", **kwargs):
        super(ProteinDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance = balance
        self.kwargs = kwargs

        self.concat = concat
        self.task_type = task_type
        self.y_type = y_type
        fuz = self.kwargs.pop('fuz', False) if self.task_type == 'RG4BP_vs_RBP' else False
        update_label = self.kwargs.pop('update_label', False)
        overwrite = self.kwargs.pop('overwrite', False)

        self.prepare = {'overwrite': overwrite, 'update_label': update_label}

        self.common_params = {
            'ESM': use_esm, 'fuz': fuz, 'loop': True,
            'k': 10, 'd_cutoff': 8.0, **self.kwargs
        }

        y = "multi" if y_type == "str" else 0
        self.all_folder = "data/uniprotkb/" 
        self.params = {
            "ALL": {"pfam_df": None, "label_path": "data/uniprotkb_binding_vectors.csv", "y": y}
        }

        if y_type != "str":
            for key in self.params:
                self.params[key]["label_path"] = None

        self.activation_path = "code/sae/activations.pt"

        self.train_dataset = self.val_dataset = self.test_dataset = None


    def prepare_data(self):
        params = {}
        common_params = {**self.common_params, **self.prepare}
        for key in self.params:
            params[key] = {**self.params[key], **common_params}
        ProteinGraph(root=self.all_folder, **params["ALL"])
        print("Data preparation completed.")


    def setup(self, stage="train_test", file_path=None, **kwargs):
        params = {}
        self.stage = stage
        common_params = self.common_params
        for key in self.params:
            params[key] = {**self.params[key], **common_params}

        datasetALL = ProteinGraph(root=self.all_folder, **params["ALL"])

        if kwargs.get("act", False):
            activations = torch.load(self.activation_path, weights_only=True)
            prot_lens = activations["prot_lens"]
            all_acts_topk = activations["acts_topk"]
            offsets = {}
            offset = 0
            for prot_id, length in prot_lens.items():
                offsets[prot_id] = (offset, offset + length)
                offset += length
            datasetALL = DatasetWrapper2(datasetALL, offsets, all_acts_topk)


        def get_file_paths(default_paths, override_path):
            return override_path if override_path is not None else default_paths 

        if self.task_type == 'pretrain':
            file_paths = get_file_paths([
                "data/uniprotkb/train.txt",
                "data/uniprotkb/valid.txt",
                "data/uniprotkb/test.txt",
            ], file_path)
            self.train_dataset, self.val_dataset, self.test_dataset = utils.split_dataset_by_uniprot_ids(datasetALL, file_paths)
            if self.balance == 'up':
                manager = ClusterDataManager(output_dir="data/uniprotkb/")
                positive_ids, _ = manager.get_ids_for_class("RNA_G4_Binding")
                positive_ids_dict = {"positives": positive_ids}
                positive_dataset = utils.split_dataset_by_ids(datasetALL, positive_ids_dict)[0]
                self.train_dataset = self.concat_datasets(self.train_dataset, positive_dataset)
                
        elif self.task_type == 'domain':
            file_paths = get_file_paths([
                "data/uniprotkb/pfam_ids.txt",
            ], file_path)

            self.train_dataset = utils.split_dataset_by_uniprot_ids(datasetALL, file_paths)[0]

        else:
            file_RBP = get_file_paths([
                "data/rbp/pdb/train/uniprot_ids.txt",
                "data/rbp/pdb/test/uniprot_ids.txt",
            ], file_path)
            # dataset_RBP, dataset_RBP_t
            dataset_RBP_train, dataset_RBP_test = utils.split_dataset_by_uniprot_ids(datasetALL, file_RBP)
            
            file_RG4BP = get_file_paths([
                "data/rg4bp/pdb/RG4_BP/train/uniprot_ids.txt",
                "data/rg4bp/pdb/RG4_BP/test/uniprot_ids.txt",
            ], file_path)
            # dataset_RG4BP, dataset_RG4BP_t
            dataset_RG4BP_train, dataset_RG4BP_test = utils.split_dataset_by_uniprot_ids(datasetALL, file_RG4BP)
            
            if self.concat:
                file_RG4BP_ALL = get_file_paths([
                "data/rg4bp/pdb/train/uniprot_ids.txt",
                "data/rg4bp/pdb/test/uniprot_ids.txt"
                ], file_path)        
                # dataset_RG4BP_all
                RG4BP_train, RG4BP_test = utils.split_dataset_by_uniprot_ids(datasetALL, file_RG4BP_ALL)
                dataset_RG4BP_train = ConcatDataset([RG4BP_train, dataset_RG4BP_train])
                dataset_RG4BP_test = ConcatDataset([RG4BP_test, dataset_RG4BP_test])
            
            if self.y_type != "str":
                dataset_RG4BP_train.data.y = torch.ones_like(dataset_RG4BP_train.data.y)
                dataset_RG4BP_test.data.y = torch.ones_like(dataset_RG4BP_test.data.y)

            if self.task_type == 'RG4BP_vs_RBP':
                self.test_dataset = ConcatDataset([dataset_RG4BP_test, dataset_RBP_test])
                self.train_dataset, self.val_dataset = self.setup_datasets(dataset_RBP_train, dataset_RG4BP_train)
            
            elif self.task_type == 'RG4BP_vs_nonNABP':
                datasetN = utils.split_dataset_by_uniprot_ids(datasetALL, ["data/HUMAN_uniprot_ids.txt"])[0]
                datasetN_train, datasetN_test = utils.split_dataset(datasetN, 'train_test', train_ratio=0.8, val_ratio=0.2)
                self.test_dataset = ConcatDataset(dataset_RG4BP_test, datasetN_test)
                self.train_dataset, self.val_dataset = self.setup_datasets(datasetN_train, dataset_RG4BP_train)

            elif self.task_type == 'RBP_vs_nonNABP':
                datasetN = utils.split_dataset_by_uniprot_ids(datasetALL, ["data/HUMAN_uniprot_ids.txt"])[0]
                if self.y_type != "str":
                    dataset_RBP_train.data.y = torch.ones_like(dataset_RBP_train.data.y)
                    dataset_RBP_test.data.y = torch.ones_like(dataset_RBP_test.data.y)
                datasetN_train, datasetN_test = utils.split_dataset(datasetN, 'train_test', train_ratio=0.8, val_ratio=0.2)
                self.test_dataset = ConcatDataset(dataset_RBP_test, datasetN_test)
                self.train_dataset, self.val_dataset = self.setup_datasets(datasetN_train, dataset_RBP_train)

        if stage=="train_test" and self.val_dataset is not None:
            self.train_dataset = self.concat_datasets(self.train_dataset, self.val_dataset)
            self.val_dataset = None




    # ==============================
    # Dataset processing
    # ==============================
    def setup_datasets(self, train1, train2):
        """
        Set up training and validation datasets.
        Args:
            train1: The first training dataset.
            train2: The second training dataset.
        Returns:
            train_dataset: The training dataset.
            val_dataset: The validation dataset (if applicable).
        """
        if self.stage == "train_test":
            train_dataset = self.concat_datasets(train1, train2)
            val_dataset = None
        elif self.stage == "train_val_test":
            train1_train, train1_val = utils.split_dataset(train1, 'train_test', train_ratio=0.9, val_ratio=0.1)
            train2_train, train2_val = utils.split_dataset(train2, 'train_test', train_ratio=0.9, val_ratio=0.1)
            train_dataset = self.concat_datasets(train1_train, train2_train)
            val_dataset = ConcatDataset([train1_val, train2_val])
        else:
            raise ValueError(f'Unsupported stage: {self.stage}. Please choose a valid stage.')

        return train_dataset, val_dataset


    def concat_datasets(self, train1, train2):
        len1 = len(train1)
        len2 = len(train2)
        total_train_num = len1 + len2

        if len1 < len2:
            train1, train2 = train2, train1
            len1, len2 = len2, len1
        if len1 == len2:
            self.balance = 'no'

        if self.balance == 'mix':
            train1 = resample(train1, replace=False, n_samples=total_train_num // 2, random_state=0)
            train2 = resample(train2, replace=True, n_samples=total_train_num // 2, random_state=0)
        elif self.balance == 'up':
            # upsample
            train2 = resample(train2, replace=True, n_samples=len1, random_state=0)
        elif self.balance == 'down':
            # downsample
            train1 = resample(train1, replace=False, n_samples=len2, random_state=0)

        return ConcatDataset([train1, train2])


    def shuffle_dataset(self, dataset, wrap=False):
        indices = torch.randperm(len(dataset))
        wrapped_dataset = Subset(dataset, indices)
        if wrap:
            wrapped_dataset = DatasetWrapper(wrapped_dataset)
        return wrapped_dataset


    def combine_dataset(self):
        """
        Returns the concatenated dataset of self.train_dataset, self.val_dataset, and self.test_dataset.
        """
        if self.train_dataset is None:
            raise RuntimeError("Dataset is not set. Call setup() first.")
        if self.stage == "train_test":
            combined_dataset = ConcatDataset([self.train_dataset, self.test_dataset])
        elif self.stage == "train_val_test":
            combined_dataset = ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])
        
        return self.shuffle_dataset(combined_dataset)
        

    # ==============================
    # Functions
    # ==============================
    def check(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            print("Datasets not initialized. Running setup...")
            self.setup('train_val_test')
        print(self.train_dataset[0])

    def count_labels(self, visualize=False):
        """Count the number of positive and negative samples in the training, validation, and test sets, along with their relative proportions."""
        # If datasets are not initialized, call setup
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            print("Datasets not initialized. Running setup...")
            self.setup('train_val_test')

        dataloaders = {
            'Train': self.train_dataloader(),
            'Validation': self.val_dataloader(),
            'Test': self.test_dataloader()
        }

        with open("data/uniprotkb_index_mapping.json", 'r') as f:
            index_mapping = json.load(f)

        def visualize_distribution(labels, counts, name):
            """Helper function to visualize data distribution."""
            # Bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=labels, y=counts)
            plt.title(f"{name} Dataset: Positive Sample Distribution")
            plt.xlabel("Labels")
            plt.ylabel("Positive Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(labels)))
            plt.title(f"{name} Dataset: Positive Sample Proportion")
            plt.show()

        stats = {}
        for name, dataloader in dataloaders.items():
            if dataloader is None:
                stats[name] = "Dataset not loaded"
                continue

            total_samples = 0
            positive_counts = None

            for batch in dataloader:
                if not hasattr(batch, 'y'):
                    raise ValueError(f"Batch in {name} does not have attribute 'y'. Check dataset format.")

                y = batch.y
                if positive_counts is None:
                    positive_counts = torch.zeros(y.shape[1], dtype=torch.int64)

                positive_counts += (y == 1).sum(dim=0)
                total_samples += y.shape[0]

            # Calculate ratios
            positive_ratios = (positive_counts.float() / total_samples).tolist() if total_samples > 0 else []

            # Store stats
            stats[name] = {
                'positive_counts': positive_counts.tolist(),
                'total_samples': total_samples,
                'positive_ratios': positive_ratios
            }

            # Prepare data for visualization
            labels = list(index_mapping.keys())
            counts = stats[name]["positive_counts"]
            ratios = stats[name]["positive_ratios"]

            # Print tabular representation
            df = pd.DataFrame({
                "Label": labels,
                "Positive Count": counts,
                "Positive Ratio": ratios
            })
            print(f"\n{name} dataset:\nTabular Representation:")
            print(df)
            
            if visualize:
                # Visualize
                visualize_distribution(labels, counts, name)

        return stats


    # ==============================
    # Set Dataloaders
    # ==============================
    def train_dataloader(self, shuffle=True):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not set. Call setup() first.")
        shuffled_train_dataset = self.shuffle_dataset(self.train_dataset)# , wrap=True
        return DataLoader(shuffled_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          shuffle=shuffle)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not set. Call setup() first.")
        self.val_dataset = self.shuffle_dataset(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          # drop_last=True
                          )
    
    def test_dataloader(self):

        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not set. Call setup() first.")
        self.test_dataset = self.shuffle_dataset(self.test_dataset)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def all_dataloader(self):
        combine_dataset = self.combine_dataset()
        return DataLoader(combine_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

if __name__ == "__main__":
    print("Start")
    data_RG4BP = ProteinDataModule(task_type='pretrain',
                                    y_type = 'str',
                                    concat = True,
                                    balance = 'no', 
                                    batch_size=16, num_workers=0,
                                    update_label=True,
                                    model_name = 'esm2_t33_650M_UR50D',
                                    layer_num = 33,
                                    hub_dir = 'code/esm2_t33_650M_UR50D',
                                    )
    data_RG4BP.prepare_data()
    data_RG4BP.check()
    data_RG4BP.count_labels()