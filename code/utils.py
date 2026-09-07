import os
import json
import torch
import GPUtil
import random
from torch.utils.data import random_split, Subset
from torch_geometric.nn import global_mean_pool


def get_default_params(alpha_dict=None):
    if alpha_dict is True:
        alpha_dict = {
            "Binding": (0.5, 0.5),
            "Nucleic Acid Binding": (1.5, 1.0),
            "only_RNA_Binding": (1.5, 1.0),
            "RNA_Binding": (2.0, 1.0),
            "all_RNA_Binding": (2.0, 1.0),
            "RNA_G4_Binding": (10.0, 1.0),
            'all_RNA_G4_Binding': (10.0, 1.0),
            # other task ...
        }
    else:
        alpha_dict = None
    
    task_names = [
        'Binding', 
        'Protein & Peptide Binding', 'Small Molecule Binding',  
        'Nucleic Acid Binding', 
        # 'add_Nucleic Acid Binding',
        # 'only_RNA_Binding',
        # 'RNA_Binding',  
        'all_RNA_Binding',  
        # 'RNA_G4_Binding',  
        'all_RNA_G4_Binding',          
    ]
    params = {
        "n_features": 1847,
        "in_channels": 1280,
        "hidden_channels": 256,
        "num_layers": 2,
        "task_names": task_names,
        "alpha_dict": alpha_dict,
        "attn_order": "domain_first",
        "use_domain": True,
        "use_sae": True,
        "head": 4,
        "lr": 3e-4,
        'betas': (0.9, 0.95),
    }

    return params


def get_available_gpu(min_memory_ratio=3/4):
    try:
        gpus = torch.cuda.get_device_properties(0)  # Assuming get_device_properties(0) returns a list-like object of GPUs
    except Exception as e:
        print(f"Error accessing GPU properties: {e}")
        gpus = []
    
    if not gpus or not torch.cuda.is_available():
        print("No GPUs found, selecting CPU.")
        return torch.device('cpu')
    
    # Get list of all GPU details
    gpus = GPUtil.getGPUs()
    
    # Filter GPUs by the minimum memory ratio
    available_gpus = [gpu for gpu in gpus if gpu.memoryFree / gpu.memoryTotal >= min_memory_ratio]
    
    if available_gpus:
        # If there are available GPUs, select the one with the most free memory
        available_gpus.sort(key=lambda gpu: gpu.memoryFree, reverse=True)
        device = torch.device(f'cuda:{available_gpus[0].id}')
        print(f"Selected cuda:{available_gpus[0].id} due to its available memory.")
    else:
        # If no GPU meets the criteria, use CPU
        device = torch.device('cpu')
        print("Selected CPU due to insufficient memory on GPU.")
    
    return device


def read_uniprot_ids(file_paths):
    """
    Read files and generate a dictionary of UniProt IDs.

    Args:
        file_paths (list): List of file paths.

    Returns:
        dict: Dictionary containing sets of UniProt IDs for each file.
    """
    uniprot_ids = {}
    with open("/home/wengnan/data/uniprotkb/clusters.json", "r") as f:
        cluster_dict = json.load(f)
    
    # Read each file, assuming the file name represents the split type, e.g., train.txt, val.txt, test.txt
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        split_type = os.path.basename(file_path).split('.')[0]  # Use the file name as the split type
        if file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                uniprot_ids[split_type] = set(line.strip() for line in f)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                cluster_ids = set(json.load(f))
                ids = [seq for cid in cluster_ids for seq in cluster_dict[cid]]
                uniprot_ids[split_type] = ids  # assumes a list of IDs
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    return uniprot_ids

def split_dataset_by_ids(dataset, uniprot_ids):
    """
    Split the dataset based on the provided UniProt IDs.

    Args:
        dataset (Dataset): Original dataset.
        uniprot_ids (dict): Dictionary containing sets of UniProt IDs for each split type.

    Returns:
        list: List of subsets of the dataset for each split type.
    """
    # Split the dataset based on the provided UniProt IDs
    split_datasets = []
    for split_type, ids in uniprot_ids.items():
        split_indices = [i for i, prot_id in enumerate(dataset.structure_ids) if prot_id in ids]
        
        if not split_indices:
            raise ValueError(f"No matching dataset entries found for provided {split_type} IDs.")
        
        split_datasets.append(Subset(dataset, split_indices))
    
    return split_datasets


def split_dataset_by_uniprot_ids(dataset, file_paths):
    """
    Split the dataset based on the provided UniProt IDs from files.

    Args:
        dataset (Dataset): Original dataset.
        file_paths (list): List of file paths containing UniProt IDs.

    Returns:
        list: List of subsets of the dataset for each split type.
    """
    # Read UniProt IDs from files
    uniprot_ids = read_uniprot_ids(file_paths)
    # Split the dataset based on the provided UniProt IDs
    return split_dataset_by_ids(dataset, uniprot_ids)


def split_dataset(dataset, split_type='train_val_test', train_ratio=0.8, val_ratio=0.1, seed=42):
    torch.manual_seed(seed)

    if not (0 <= train_ratio <= 1):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0 <= val_ratio <= 1):
        raise ValueError("val_ratio must be between 0 and 1")
    if split_type == 'train_val_test' and (train_ratio + val_ratio >= 1):
        raise ValueError("The sum of train_ratio and val_ratio must be less than 1 for 'train_val_test' split_type")

    if split_type == 'train_val_test':
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
    
    elif split_type == 'train_test':
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset
    
    else:
        raise ValueError("split_type should be either 'train_val_test' or 'train_test'")
    

def split_dataset_2(dataset, val_size, seed=42):
    torch.manual_seed(seed)

    if not (0 <= val_size <= len(dataset)):
        raise ValueError("val_size must be between 0 and the length of the dataset")

    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
    

def hextract_features(dataloader, model, device):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch.to(device)
            batch_size = batch.y.size(0)
            batch_concat = []
            for i in range(batch_size):  
                num_nodes = len(batch[i].x_dict['protein'])
                batch_tensor = torch.full((num_nodes,), i, dtype=torch.long)
                batch_concat.append(batch_tensor)
            batch_concat = torch.cat(batch_concat)
            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            # Heterogeneous graphs may also include edge attributes, check if present
            edge_attr_dict = getattr(batch, 'edge_attr_dict', None)
            # Pass the appropriate arguments to the gnn_model
            embeddings = model(x_dict, edge_index_dict, edge_attr_dict)
            protein_embeddings = embeddings['protein']
            batch_concat = batch_concat.to(device)
            embeddings = global_mean_pool(protein_embeddings, batch_concat)
            all_embeddings.append(embeddings)
            all_labels.append(batch.y)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def extract_features(dataloader, model, device):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = getattr(batch, 'edge_attr', None)
            annotations = getattr(batch, 'annotations', None)
            # Pass the appropriate arguments to the gnn_model
            embeddings = model(x, edge_index, edge_attr, annotations)
            embeddings = global_mean_pool(embeddings, batch.batch)
            all_embeddings.append(embeddings)
            all_labels.append(batch.y)
    return torch.cat(all_embeddings), torch.cat(all_labels)


# Downsample datasetN to match the size of dataset_RG4BP and dataset_RBP
def downsample_dataset(dataset, target_size, seed=42):
    """Downsamples a dataset to a target size."""
    random.seed(seed)
    indices = random.sample(range(len(dataset)), target_size)
    return Subset(dataset, indices)