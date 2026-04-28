import os
import re
import torch
import csv
import esm
import json
import utils
import freesasa
import residue_properties
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import is_aa
from propy import PyPro
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import add_self_loops
from transformers import AutoTokenizer, EsmModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

os.chdir(os.path.expanduser("~"))


class RigidTransform:
    """
    Represents a rigid transformation in 3D space.

    Args:
        t (torch.Tensor): The translation vector of shape (..., 3).
        R (torch.Tensor, optional): The rotation matrix of shape (..., 3, 3). If not provided, a zero matrix will be used.

    Attributes:
        t (torch.Tensor): The translation vector.
        R (torch.Tensor): The rotation matrix.

    Methods:
        __getitem__(self, key): Returns a new RigidTransform object with sliced translation and rotation.
        unsqueeze(self, dim): Returns a new RigidTransform object with unsqueezed translation and rotation.
        squeeze(self, dim): Returns a new RigidTransform object with squeezed translation and rotation.
        concatenate(self, other, dim=0): Concatenates two RigidTransform objects along the specified dimension.

    """

    def __init__(self, t: torch.Tensor, R: torch.Tensor = None):
        self.t = t
        if R is None:
            R = t.new_zeros(*t.shape, 3)
        self.R = R

    def __getitem__(self, key):
        return RigidTransform(self.t[key], self.R[key])

    def unsqueeze(self, dim):
        return RigidTransform(self.t.unsqueeze(dim), self.R.unsqueeze(dim))

    def squeeze(self, dim):
        return RigidTransform(self.t.squeeze(dim), self.R.squeeze(dim))

    def concatenate(self, other, dim=0):
        return RigidTransform(
            torch.cat([self.t, other.t], dim=dim),
            torch.cat([self.R, other.R], dim=dim),
        )


def get_frame_matrix(
    ri: torch.Tensor, rj: torch.Tensor, rk: torch.Tensor, eps=1e-4, strict=False
):
    """
    Compute the frame matrix given three input vectors.

    Args:
        ri (torch.Tensor): The first input vector.
        rj (torch.Tensor): The second input vector.
        rk (torch.Tensor): The third input vector.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-4.
        strict (bool, optional): If True, perform strict Gram-Schmidt orthogonalization. If False, perform modified Gram-Schmidt orthogonalization. Defaults to False.

    Returns:
        torch.Tensor: The frame matrix.

    """
    # Regularized Gram-Schmidt
    # Allow for shearing
    v1 = ri - rj
    v2 = rk - rj
    if strict:
        # v1 = v1 + torch.randn_like(rj).mul(eps)
        # v2 = v2 + torch.randn_like(rj).mul(eps)
        e1 = v1 / v1.norm(dim=-1, keepdim=True)
        # Project and pad
        u2 = v2 - e1.mul(e1.mul(v2).sum(-1, keepdim=True))
        e2 = u2 / u2.norm(dim=-1, keepdim=True)
    else:
        e1 = v1 / v1.square().sum(dim=-1, keepdim=True).add(eps).sqrt()
        # Project and pad
        u2 = v2 - e1.mul(e1.mul(v2).sum(-1, keepdim=True))
        e2 = u2 / u2.square().sum(dim=-1, keepdim=True).add(eps).sqrt()
    e3 = torch.cross(e1, e2, dim=-1)
    # Rows - lab frame, columns - internal frame
    rot_j = torch.stack([e1, e2, e3], dim=-1)
    return RigidTransform(rj, torch.nan_to_num(rot_j, 0.0))


def gaussian_rbf_encoding1d(x: torch.Tensor, n_basis, x_max, sigma=0.8):
    """
    Encodes a 1-dimensional tensor using Gaussian Radial Basis Function (RBF) encoding.

    Args:
        x (torch.Tensor): The input tensor to be encoded.
        n_basis (int): The number of basis functions to use for encoding.
        x_max (float): The maximum value of x.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1.0.

    Returns:
        torch.Tensor: The encoded tensor.

    """
    rbf_centers = torch.linspace(0, x_max, n_basis)
    encodings = torch.exp(
        -((x.unsqueeze(-1) - rbf_centers).div(sigma).square())
    )
    return encodings


def position_encoding_matrix(max_len, d_model):
    """
    Generate a position encoding matrix for each position in the sequence.
    
    Args:
    max_len (int): The maximum length of the sequence.
    d_model (int): The embedding dimension.
    
    Returns:
    numpy.ndarray: The position encoding matrix.
    """
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    position_encodings = np.zeros((max_len, d_model))
    
    for pos in range(max_len):
        angle_rads = pos * angle_rates
        angle_rads[0::2] = np.sin(angle_rads[0::2])
        angle_rads[1::2] = np.cos(angle_rads[1::2])
        position_encodings[pos] = angle_rads
    
    return position_encodings


def relative_geometry_encoding(atom_positions, merged_edge_idx, n_basis=10, d_max=8.0):
    
    # start_indices = merged_edge_idx[0].numpy()
    # end_indices = merged_edge_idx[1].numpy()
    # start_encodings = position_encodings[start_indices]
    # end_encodings = position_encodings[end_indices]
    # pos_encoding = np.abs(start_encodings - end_encodings)
    # pos_encoding = torch.tensor(pos_encoding, dtype=torch.float)

    # Convert the atom positions to a tensor
    n_residues = len(atom_positions)
    n_atoms = next(iter(atom_positions.values())).shape[0]
    atom_positions_tensor = torch.zeros((n_residues, n_atoms, 3))
    for i, residue_id in enumerate(sorted(atom_positions.keys())):
        atom_positions_tensor[i] = torch.from_numpy(atom_positions[residue_id])

    # Check that each residue has at least three atoms
    assert n_atoms >= 3, "Each residue must have at least three atoms."

    # Get the backbone frames
    frames = get_frame_matrix(
        atom_positions_tensor[:, 0, :],
        atom_positions_tensor[:, 1, :],
        atom_positions_tensor[:, 2, :],
    )

    frame_t, frame_R = frames.t, frames.R
    pair_dists = torch.norm(
        frame_t[merged_edge_idx[0]] - frame_t[merged_edge_idx[1]],
        dim=-1,
    )
    pair_directions_l = torch.matmul(
        (frame_t[merged_edge_idx[1]] - frame_t[merged_edge_idx[0]]).unsqueeze(-2),
        frame_R[merged_edge_idx[0]],
    ).squeeze(-2) / pair_dists.square().add(1).sqrt().unsqueeze(-1)
    pair_directions_r = torch.matmul(
        (frame_t[merged_edge_idx[0]] - frame_t[merged_edge_idx[1]]).unsqueeze(-2),
        frame_R[merged_edge_idx[1]],
    ).squeeze(-2) / pair_dists.square().add(1).sqrt().unsqueeze(-1)
    pair_orientations = torch.matmul(
        frame_R.transpose(-2, -1).contiguous()[merged_edge_idx[0]],
        frame_R[merged_edge_idx[1]],
    )

    geo_encoding = torch.cat([
            gaussian_rbf_encoding1d(pair_dists, n_basis, d_max),
            pair_directions_l,
            pair_directions_r,
            pair_orientations.flatten(-2, -1),
        ],
        dim=-1,
    )

    # encoding = torch.cat([pos_encoding, geo_encoding], dim=-1)

    return geo_encoding


def onehot_to_sequence(onehot_matrix, residue_id2symbol):
    """
    Convert the one-hot encoding matrix into an amino acid sequence.
    
    Args:
        onehot_matrix (np.ndarray): A 2D matrix where each row represents the one-hot encoding of a residue.
        residue_id2symbol (dict): The inverse mapping of residue_symbol2id, with keys as numerical indices and values as single-letter amino acid codes.
        
    Returns:
        str: An amino acid sequence.
    """
    sequence = []
    for onehot in onehot_matrix:
        idx = int(np.argmax(onehot))  
        amino_acid = residue_id2symbol.get(idx, 'X')  
        sequence.append(amino_acid)
    return ''.join(sequence)



class PDB2Graph:
    # ==============================
    # Global Parameters
    # ==============================

    column_names =['Sequence ID', 'MD5 Checksum', 'Sequence Length', 'Analysis Method', 
            'Accession Number', 'Domain Name', 'Start Position', 'End Position', 
            'E-value', 'Status', 'Date', 'InterPro Entry ID', 'InterPro Entry Description', 
            'GO Terms', 'Pathway Entry']
    
    residue_symbol2id = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3,
        'C': 4, 'Q': 5, 'E': 6, 'G': 7,
        'H': 8, 'I': 9, 'L': 10, 'K': 11,
        'M': 12, 'F': 13, 'P': 14, 'S': 15,
        'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    
    atom_types = [
        "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2",
        "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3",
        "NZ", "OXT",
    ]
    atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
    atom_type_num = len(atom_types)  # := 37.
    element_id = {"C": 6, "N": 7, "O": 8, "S": 16,}
    device = utils.get_available_gpu()


    def __three_to_one(self, res_name):
        mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return mapping.get(res_name, 'X')  # 'X' for unknown residues
    
    # position_encodings = position_encoding_matrix(max_len=15000, d_model=30)


    # ==============================
    # Initialization and Setup
    # ==============================
    def __init__(self, pfam_path=None, label_path=None, normalize=False, 
                 onehot=False, **kwargs):# , update=False, model_RG4BP=None, model_RBP=None
        
        model_name = kwargs.get("model_name", 'esm2_t6_8M_UR50D')
        self.layer_num = kwargs.get("layer_num", 6)
        hub_dir = kwargs.get("hub_dir", 'code/esm2_t6_8M_UR50D')
        torch.hub.set_dir(hub_dir)
        model_esm, alphabet = self.__load_esm_model(model_name)
        self.batch_converter = alphabet.get_batch_converter() 
        self.model_esm = model_esm.to(self.device)
        self.model_esm.eval()

        self.fuz = kwargs.pop('fuz', False)
        self.molecule_features = kwargs.pop('molecule_features', False)
        if pfam_path is None:
            self.pfam_df = None
        elif not os.path.exists(pfam_path):
            raise ValueError("Invalid or missing pfam_path")
        else:
            pfam_df = pd.read_csv(pfam_path, sep='\t', names=self.column_names)
            pfam_df['UniprotID'] = pfam_df['Sequence ID'].apply(lambda x: x.split('|')[1])
            self.pfam_df = pfam_df
            base_dir = os.path.dirname(pfam_path)
            self.fuzdrop_dir = os.path.join(base_dir, "fuzdrop/")
            llps_file_path = os.path.join(base_dir, "llps.npy")
            if self.fuz:
                if not os.path.exists(self.fuzdrop_dir):
                    raise ValueError(f"Directory does not exist: {self.fuzdrop_dir}")
                if not os.path.isfile(llps_file_path):
                    raise ValueError(f"File does not exist: {llps_file_path}")
                self.llps_data = np.load(llps_file_path, allow_pickle=True).item()
        
        if label_path is None:
            self.labels = False
        else:
            self.labels = True
            self.protein_labels = {}
            label_path = os.path.join(os.path.expanduser("~"), label_path)
            with open(label_path, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    uniprot_id = row["UniProt_ID"]
                    binding_vector = json.loads(row["Binding_Vector_JSON"])
                    self.protein_labels[uniprot_id] = binding_vector

        self.normalize = normalize
        self.onehot = onehot
        attributes = ['structure', 'sequence', 'prot_idx', 'coordinates', 'cls_embedding']
        for attr in attributes:
            setattr(self, attr, None)

        self.__merge_residue_properties()
        if self.normalize:  
            self.__normalize_residue_properties()

    def __load_esm_model(self, model_name):
        """
        Load different ESM-2 pretrained models based on the provided model name.

        Args:
        - model_name (str): The name of the ESM-2 model.

        Returns:
        - model: The loaded ESM-2 pretrained model.
        - alphabet: The corresponding alphabet.
        """
        # Ensure the model name matches the naming in `esm.pretrained`
        model_func = getattr(esm.pretrained, model_name, None)
        if model_func is None:
            raise ValueError(f"ESM pretrained model '{model_name}' does not exist. Please check the model name.")

        # Load the model
        return model_func()
    

    def __parse_pdb(self, file_path):
        """
        Parses a PDB file and returns the structure and structure ID.

        Args:
            file_path (str): The path to the PDB file.

        Returns:
            structure (Bio.PDB.Structure): The parsed structure object.
            structure_id (str): The ID of the structure.

        """
        self.prot_idx = re.search(r'AF-(.*?)-F1', file_path).group(1)
        self.file_path = file_path
        parser = PDB.PDBParser(PERMISSIVE=1)
        self.structure = parser.get_structure(self.prot_idx, file_path)
        self.sequence = ""
        self.coordinates = {}
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    res_name = self.__three_to_one(residue.get_resname())
                    self.sequence += res_name
                    if 'CA' in residue:
                        res_id = residue.get_id()
                        coor = residue['CA'].get_vector().get_array()
                        self.coordinates[res_id] = coor


    def __normalize_residue_properties(self):
        """
        Normalize continuous features in self.residue_properties while keeping binary features unchanged.

        Returns:
            dict: The updated residue_properties dictionary with normalized continuous features.
        """
        # Specify binary features (one-hot like)
        binary_features = ['polarity', 'charge', 'hydrophobic', 'aromatic']

        # Extract all feature names (binary + continuous)
        feature_names = list(next(iter(self.residue_properties.values())).keys())
        continuous_features = [feat for feat in feature_names if feat not in binary_features]

        # Prepare data for normalization
        residue_list = list(self.residue_properties.keys())
        continuous_data = []

        for residue in residue_list:
            continuous_data.append([self.residue_properties[residue][feat] for feat in continuous_features])

        # Normalize continuous features using MinMaxScaler or StandardScaler
        scaler = MinMaxScaler()  # You can also use StandardScaler()
        continuous_data_normalized = scaler.fit_transform(continuous_data)

        # Update the residue_properties dictionary
        for idx, residue in enumerate(residue_list):
            for i, feat in enumerate(continuous_features):
                self.residue_properties[residue][feat] = continuous_data_normalized[idx][i]

        return self.residue_properties


    def __merge_residue_properties(self):
        """
        Merge various residue property dictionaries into the residue_properties dictionary.

        Returns:
            dict: The updated residue_properties dictionary with additional properties.
        """
        self.residue_properties = residue_properties.residue_properties
        additional_properties = {
            'NAB_mRNAinteractome': residue_properties.NAB_mRNAinteractome,
            'NAB_classicalRBD': residue_properties.NAB_classicalRBD,
            'NAB_nonclassicalRBD': residue_properties.NAB_nonclassicalRBD,
            'NAB_unknownRBD': residue_properties.NAB_unknownRBD,
            'alpha_helix_Palau': residue_properties.alpha_helix_Palau,
            'Burial_Radzicka': residue_properties.Burial_Radzicka,
            'Burial_Janin': residue_properties.Burial_Janin,
            'Burial_Chothia': residue_properties.Burial_Chothia,
            'membrane_Punta1': residue_properties.membrane_Punta1,
            'membrane_Punta2': residue_properties.membrane_Punta2,
            'hydrophobicity_Eisenberg': residue_properties.hydrophobicity_Eisenberg,
            'hydrophobicity_Janin': residue_properties.hydrophobicity_Janin,
            'hydrophobicity_Fauchere': residue_properties.hydrophobicity_Fauchere,
            'hydrophobicity_Leo': residue_properties.hydrophobicity_Leo,
        }

        for residue, properties in self.residue_properties.items():
            for prop_name, prop_dict in additional_properties.items():
                properties[prop_name] = prop_dict.get(residue, 0.0)

        return self.residue_properties

        
    # ==============================
    # Molecule Features
    # ==============================
    def __molecule_features(self):
        DesObject = PyPro.GetProDes(self.sequence)
        # Extract CTD features
        ctd_features = DesObject.GetCTD()
        ctd_features = np.array(list(ctd_features.values()))
        # # print(ctd_features)
        # normalized_ctd = self.normalize_features(ctd_features)
        # print(normalized_ctd.shape)
        # Extract AAC features
        aac_features = DesObject.GetAAComp()
        aac_features = np.array(list(aac_features.values()))
        # normalized_aac = self.normalize_features(aac_features)

        # Extract DPC features
        dpc_features = DesObject.GetDPComp()
        dpc_features = np.array(list(dpc_features.values()))
        # normalized_dpc = self.normalize_features(dpc_features)

        # Concatenate all features into a single feature vector
        feature_vector = np.concatenate([ctd_features, aac_features, dpc_features])
        # feature_vector = np.concatenate([normalized_ctd, normalized_aac, normalized_dpc])
        feature_vector = feature_vector.reshape(1, -1)
        return feature_vector


    # ==============================
    # Residue Features
    # ==============================
    def __residue_features(self, ESM=False, **kwargs):
        """
        Function to get node features from a protein structure.

        Args:
            ESM (bool, optional): Whether to use ESM-2 model for feature extraction. Defaults to False.
            **kwargs: Additional keyword arguments for feature extraction.

        Returns:
            torch.Tensor: A tensor containing the node features.

        """
        node_features = []
        x = None
        cls_embedding = torch.zeros(1, 1)

        if ESM:# 320
            embeddings = self.__extract_residue_embeddings()
            node_embeddings = embeddings[:, 1:-1, :].squeeze(0)  # Remove padding nodes at the beginning and end
            x = node_embeddings if x is None else torch.cat([x, node_embeddings], dim=1)
            cls_embedding = embeddings[:, 0, :].squeeze(-1) # Extract the embedding of the [CLS] token

        else:
            residue_types = self.__extract_residue_types()
            node_features.append({'residue_types': residue_types}) # 21 or 1

            chem_features = self.__get_chem_features() 
            node_features.append({'chem_features': chem_features}) # 23

        # Check if parameter is provided and set to True
        if self.fuz:
            fuz_features = self.__get_pDP_Sbind_LLPs() 
            node_features.append({'fuz_features': fuz_features}) # 2

        if len(node_features) != 0:
            for feature_dict in node_features:
                for key, feature in feature_dict.items():
                    # Concatenate other features
                    feature_tensor = torch.tensor(feature, dtype=torch.float32)
                    if feature_tensor.dim() == 1:
                        feature_tensor = feature_tensor.unsqueeze(1)
                    x = feature_tensor if x is None else torch.cat([x, feature_tensor], dim=1)
            
        return x, cls_embedding


    def __get_chem_features(self, use_chirality=True):

        residue_features = []
        hse_values = self.__calculate_hse()
        sasa_values = self.__calculate_sasa()

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True):  # Processes only standard amino acids
                        res_id = residue.get_id()
                        feature_vector = self.__get_chem_feature_vector(residue) # 19
                        
                        # Optional: Chiral coding (CIP code)
                        if use_chirality:
                            chirality_vector = self.__get_residue_chirality(residue) # 2
                            feature_vector.extend(chirality_vector)
                        
                        hse_value = hse_values.get(res_id, -1) # 1
                        feature_vector.append(hse_value)
                        sasa_value = sasa_values.get(res_id, 0.0) # 1
                        feature_vector.append(sasa_value)
                        
                        residue_features.append(feature_vector)
        return np.array(residue_features) # 23


    def __get_residue_chirality(self, residue):
        """
        Specifies the chirality type of an amino acid. The default is L-type amino acids, and the R/S configuration is distinguished based on the amino acid type.
        
        Args:
            residue (Bio.PDB.Residue): An amino acid residue.
        
        Returns:
            chirality_vector: A vector containing D/L type and R/S configuration.
        """
        chirality_vector = [0, 0] 
        
        if residue.has_id('CA') and residue.get_resname() != 'GLY':
            chirality_vector[0] = 1  # Type L
            
            if residue.get_resname() == 'CYS':
                chirality_vector[1] = 1  # Type R
            else:
                chirality_vector[1] = 0  # Type S
        
        return chirality_vector
        

    def __get_chem_feature_vector(self, residue):
        resname = self.__three_to_one(residue.get_resname())
        if resname in self.residue_properties:
            props = self.residue_properties[resname]
            feature_vector = [
                props.get('polarity', 0.0),   
                props.get('charge', 0.0),     
                props.get('hydrophobic', 0.0),
                props.get('aromatic', 0.0),   
                props.get('mass', 0.0),       
                props.get('volume', 0.0),     
                props.get('NAB_mRNAinteractome', 0.0),
                props.get('NAB_classicalRBD', 0.0),
                props.get('NAB_nonclassicalRBD', 0.0),
                props.get('NAB_unknownRBD', 0.0),
                props.get('alpha_helix_Palau', 0.0),
                props.get('Burial_Radzicka', 0.0),
                props.get('Burial_Janin', 0.0),
                props.get('Burial_Chothia', 0.0),
                props.get('membrane_Punta1', 0.0),
                props.get('membrane_Punta2', 0.0),
                props.get('hydrophobicity_Eisenberg', 0.0),
                props.get('hydrophobicity_Janin', 0.0),
                props.get('hydrophobicity_Fauchere', 0.0),
                props.get('hydrophobicity_Leo', 0.0),
            ]
        else:
            # If the residue is not in the predefined dictionary, pad it with zeros
            feature_vector = [0.0] * 19  

        return feature_vector

    
    def __normalize_features(self, feature_dict, placeholder_value=-1):
        """
        Normalize the values in a feature dictionary using MinMaxScaler while handling missing values.

        Args:
            feature_dict (dict): Dictionary where keys are residue IDs and values are feature values.
                                Feature values can be scalars (int, float) or tuples (e.g., HSE values).
            placeholder_value (int, float, tuple): Value used to represent missing or invalid data. Default is -1.

        Returns:
            dict: Normalized feature dictionary where values are within [0, 1].
        """

        # Extract valid numerical values, excluding placeholder
        valid_values = [
            value for value in feature_dict.values()
            if isinstance(value, (int, float)) and value != placeholder_value
        ]
        
        if not valid_values:
            # If no valid values, return original dictionary
            return feature_dict

        # Reshape data for MinMaxScaler
        valid_values = np.array(valid_values).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(valid_values)

        def normalize(value):
            """Normalize a single value or tuple."""
            if isinstance(value, (int, float)):
                return scaler.transform([[value]])[0][0] if value != placeholder_value else placeholder_value
            elif isinstance(value, tuple):
                return tuple(
                    scaler.transform([[v]])[0][0] if v != placeholder_value else v
                    for v in value
                )
            else:
                return value

        # Apply normalization to all values in the dictionary
        return {key: normalize(value) for key, value in feature_dict.items()}


    def __calculate_hse(self, type='CN'): 
        '''
        Calculate the relative solvent accessible surface area (SASA) for C-alpha or C-beta atoms of amino acid residues in a protein structure.

        Args:
            structure (Bio.PDB.Structure): The structure object of the protein.
            type (str): The type of atoms to calculate SASA for. Choose from 'CA' or 'CB'.

        Returns:
            dict: A dictionary mapping residue IDs to their corresponding SASA values.

        Raises:
            ValueError: If an invalid type is provided.

        '''
        residue_ids = [residue.get_id() for model in self.structure for chain in model for residue in chain if PDB.is_aa(residue)]

        # Iterate over models
        for model in self.structure:
            if type == 'CA': 
                fun_name = "HSExposureCA"
                params = (model,12.0,0)
                placeholder_value = (-1,-1,-1)
            elif type == 'CB': 
                fun_name = "HSExposureCB"
                params = (model,12.0,0)
                placeholder_value = (-1,-1,-1)
            elif type == 'CN': 
                fun_name = "ExposureCN"
                params = (model,12.0,0)
                placeholder_value = -1
            else:
                raise ValueError("Invalid HSE type. Choose from 'CA', 'CB', or 'CN'.")
            
            hse = eval("PDB.HSExposure."+fun_name)(*params)
            hse_results = {
                residue[1]: exposure
                for residue, exposure in hse.property_dict.items()
            }
            
            existing_keys = set(hse_results.keys())
            missing_keys = set(residue_ids) - existing_keys
            for missing_key in missing_keys:
                hse_results[missing_key] = placeholder_value   

        # Extract only numerical values for normalization
        if self.normalize:
            hse_results = self.__normalize_features(hse_results, placeholder_value)
        
        return hse_results
    

    def __calculate_sasa(self): 

        structure_freesasa = freesasa.Structure(self.file_path)
        result = freesasa.calc(structure_freesasa)
        residue_sasa = {}

        # Iterate over models
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    res_id = residue.get_id()
                    sasa = 0.0
                    # atom_count = 0
                    for atom in residue:
                        atom_index = atom.serial_number - 1
                        sasa += result.atomArea(atom_index)
                        # atom_count += 1
                    # if atom_count > 0:
                    #     sasa /= atom_count
                    residue_sasa[res_id] =sasa
        
        if self.normalize:
            residue_sasa = self.__normalize_features(residue_sasa, placeholder_value=-1)  
        return residue_sasa
    

    def __get_pDP_Sbind_LLPs(self):
        """
        Extract pDP, Sbind, and LLPs features from a TSV file and return them as NumPy arrays.
        
        Returns:
            np.array: A 2D NumPy array where each row corresponds to [pDP, Sbind, LLPs] values for a residue.
        """
        file_path = os.path.join(self.fuzdrop_dir, f"FuzDrop-{self.prot_idx}.tsv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        if self.prot_idx not in self.llps_data:
            raise KeyError(f"LLPS value for {self.prot_idx} not found")
        
        df = pd.read_csv(file_path, sep='\t')
        self.llps_value = self.llps_data[self.prot_idx]

        # Initialize the feature matrix
        pDP_Sbind = {}

        # Iterate through the DataFrame and populate the feature matrix
        for _, row in df.iterrows():
            res_id = row['position']
            pDP = row['pDP']
            Sbind = row['Sbind']
            pDP_Sbind[res_id] = [pDP, Sbind]

        if self.normalize:
            pDP_Sbind = self.__normalize_features(pDP_Sbind, placeholder_value=-1)

        feature_list = []
        for res_id in sorted(pDP_Sbind.keys()):
            feature_list.append(pDP_Sbind[res_id])

        return np.array(feature_list)


    def __extract_residue_types(self):
        """
        Extract residue features, including one-hot encoding of residue types, from a PDB file.
        
        Args:
            structure (Bio.PDB.Structure): The structure object of the PDB file.
            
        Returns:
            residue_features: A dictionary containing residue features, where the keys are residue IDs and the values are tensors
                            representing the one-hot encoding of residue types.
        """
        residue_features = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    res_name = self.__three_to_one(residue.get_resname())
                    # Map residue name to numeric identifier using residue_symbol2id
                    idx = self.residue_symbol2id.get(res_name, 20)  # Use 20 for unknown residues
                    if self.onehot:
                        # Convert residue type to one-hot encoding
                        one_hot = torch.zeros(len(self.residue_symbol2id) + 1, dtype=torch.float32)  # Add one dimension for unknown residues
                        if idx != 20:
                            one_hot[idx] = 1
                        else:
                            one_hot[-1] = 1  # Set the last dimension to 1 for unknown residues
                        residue_features.append(one_hot.numpy())  # Convert to NumPy for concatenation
                    else:
                        residue_features.append(idx)    
        return np.array(residue_features)
    

    def __extract_residue_embeddings(self):
        """
        Extract residue embeddings from a PDB structure using the ESM-2 model.

        Args:
            structure (Bio.PDB.Structure): The structure object of the PDB file.

        Returns:
            torch.Tensor: The residue embeddings as a tensor of shape (num_residues, embedding_dim).

        """

        data = [(self.structure.get_id(), self.sequence)]

        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Generate embeddings using the ESM-2 model
        with torch.no_grad():
            results = self.model_esm(batch_tokens, repr_layers=[self.layer_num])

        # Extract embeddings from the layer
        embeddings = results["representations"][self.layer_num].to('cpu')
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

        return embeddings
    

    def extract_residue_embeddings(self, file_path):
        """
        Extract residue embeddings from a PDB structure using the ESM-2 model.

        Args:
            structure (Bio.PDB.Structure): The structure object of the PDB file.

        Returns:
            torch.Tensor: The residue embeddings as a tensor of shape (num_residues, embedding_dim).

        """
        self.__parse_pdb(file_path)

        data = [(self.structure.get_id(), self.sequence)]

        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Generate embeddings using the ESM-2 model
        with torch.no_grad():
            results = self.model_esm(batch_tokens, repr_layers=[self.layer_num])

        # Extract embeddings from the layer
        embeddings = results["representations"][self.layer_num].to('cpu')
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

        return embeddings

    
    # ==============================
    # Edge Features
    # ==============================
    def __extract_atom(self):
        """
        Extracts atom types from a Bio.PDB Structure object.
        Extracts the atom positions from a Bio.PDB Structure object.
        
        Parameters:
        - structure: A Bio.PDB Structure object containing the protein structure.
        
        Returns:
        - atom_types: A dictionary containing atom types for each atom in the structure.
        - atom_positions: A dictionary containing atom positions for each atom in the structure.
        The keys are residue IDs and the values are numpy arrays of atom types.
        """

        atom_types_ = {}
        atom_positions = {}

        for residue in self.structure.get_residues():
                eletypes = np.zeros((self.atom_type_num,))
                pos = np.zeros((self.atom_type_num, 3))
                # Iterate over all the atoms in the residue
                for atom in residue:
                    if atom.name not in self.atom_types:
                        continue
                    if atom.occupancy and atom.occupancy < 0.5:
                        # Remove too-flexible atoms
                        continue
                    # Get the atom type name (e.g., 'CA', 'CB', 'O', etc.)
                    eletypes[self.atom_order[atom.name]] = self.element_id[atom.element]
                    pos[self.atom_order[atom.name]] = atom.get_coord()
                atom_types_[residue.get_id()] = np.array(eletypes)
                atom_positions[residue.get_id()] = np.array(pos)
        
        return atom_types_, atom_positions


    def __extract_neighborhood(self, d_cutoff=8.0, k=-1, seq=False):
        """
        Extract the neighborhood information from the protein structure.

        Args:
            chain_starts (list): A list of starting residue numbers for each chain.
            d_cutoff (float, optional): The distance cutoff for defining edges. Defaults to 8.0.
            k (int, optional): The number of nearest neighbors to consider. Defaults to -1.
            seq (bool, optional): Whether to include sequence edges. Defaults to False.

        Returns:
            tuple: Three tuples, each containing the edge index and edge type of a type of edge.

        """
        edge_index_seq = edge_index_dist = edge_index_knn = None

        if seq:
            # Get the length of the sequence
            seq_len = len(self.sequence)
            # Create sequence edges
            seq_edges = torch.stack([torch.arange(seq_len - 1), torch.arange(1, seq_len)], dim=0)
            # Create reverse sequence edges
            reverse_seq_edges = seq_edges.flip(0)
            # Combine sequence edges and reverse sequence edges
            edge_index_seq = torch.cat((seq_edges, reverse_seq_edges), dim=1)
            
        pos = torch.FloatTensor([list(coord) for coord in self.coordinates.values()])
        # Calculate the distances between coordinates
        distances = torch.cdist(pos, pos)

        # Set the distance threshold
        edge_indices = torch.nonzero(distances <= d_cutoff, as_tuple=False).t()
            
        # Create the edge index of the graph, excluding self-loops and undirected edges
        dist_edge = edge_indices[:, edge_indices[0] != edge_indices[1]]
        # Create reverse edges
        reverse_dist_edge = dist_edge.flip(0)
        # Concatenate edge_index_dist and reverse_edge_index_dist along the column dimension
        edge_index_dist = torch.cat((dist_edge, reverse_dist_edge), dim=1)
        
        if k > 0:
            # Calculate k-nearest neighbor edges
            edge_index_knn = knn_graph(pos, k=k)

        return edge_index_seq, edge_index_dist, edge_index_knn



    # ==============================
    # Feature Extraction Methods
    # ==============================
    def __feature_extraction(self, ESM=False, **kwargs):
        """
        Function to extract features from a protein structure.

        Args:
            structure (Bio.PDB.Structure): The structure object of the protein.
            ESM (bool, optional): Whether to use ESM-2 model for feature extraction. Defaults to False.
            **kwargs: Additional keyword arguments for feature extraction.

        Returns:
            dict: A dictionary containing the extracted features.

        """
        features = {}
        features['atom_types'], features['atom_positions'] = self.__extract_atom()
        features['residue_features'], cls_embedding = self.__residue_features(ESM, **kwargs)

        if self.molecule_features:
            molecule_features = torch.tensor(self.__molecule_features(), dtype=torch.float32)
            features['molecule_features']= torch.cat((molecule_features, cls_embedding), dim=1)
        return features

    
    # ==============================
    # Other Functions
    # ==============================
    def __extract_and_annotate_pfam_fragments(self):

        # Get the sequence length
        sequence_length = len(self.sequence)
        
        # Initialize node features with zeros
        node_features = torch.zeros((sequence_length, 1))

        if self.pfam_df is None:
            return node_features
        else:
            # Filter the pfam_df to get annotations for the current protein
            pfam_annotations = self.pfam_df[self.pfam_df['UniprotID'] == self.prot_idx]
        
        if pfam_annotations.empty:
            return node_features
        else:
            # Iterate through the annotations and set the corresponding node features to 1 for the annotated regions
            for _, row in pfam_annotations.iterrows():
                start = row['Start Position'] - 1
                end = row['End Position']
                node_features[start:end] = 1
        
        return node_features
    

    def __add_labels(self):
        if self.prot_idx in self.protein_labels:
            y = self.protein_labels[self.prot_idx]
            y = torch.tensor(y).unsqueeze(0)
        else:
            y = torch.zeros(6).unsqueeze(0)
            # raise ValueError(f"UniProt_ID {self.prot_idx} not found in protein labels")
        return y


    def __add_edge_attr(self, x, edge_index, edge_attr, esm_use=False):

        node_edge_features = torch.zeros(x.size(0), edge_attr.size(1))  # [230, 60]

        # Iterate through each edge and add the edge's attributes to the corresponding node.
        for i in range(edge_index.size(1)):  
            node_1, node_2 = edge_index[:, i]  
            node_edge_features[node_1] += edge_attr[i]  
            node_edge_features[node_2] += edge_attr[i]  

        # if esm_use:
        #     x_augmented = torch.cat([x[:, :320], node_edge_features, x[:, 320:]], dim=1)
        # else:
        #     x_augmented = torch.cat([node_edge_features, x], dim=1) 

        x_augmented = torch.cat([x, node_edge_features], dim=1)

        return x_augmented

    
    def __update_x(self, data):
        if self.model_RG4BP is not None and self.model_RBP is not None:
            data = data.to(self.device)

            # Extract the last three dimensions from the node features as additional features
            extra_features = data.x[:, -3:]

            # Use the RG4BP and RBP models to update the node features
            updated_features_RG4BP = self.model_RG4BP.update_node_features(data)
            updated_features_RBP = self.model_RBP.update_node_features(data)

            # Concatenate the updated features from both models with the additional features
            updated_features = torch.cat([updated_features_RG4BP, updated_features_RBP, extra_features], dim=-1)

            # Update the node features in the data object
            data.x = updated_features
            return data
        else:
            return data


    def __read_chain_ids(self):
        """
        Read the chain IDs and starting residue numbers from a structure.

        Args:
            structure (Bio.PDB.Structure): The structure object containing models and chains.

        Returns:
            tuple: A tuple containing three elements:
                - chain_ids (list): A list of chain IDs.
                - chain_starts (list): A list of starting residue numbers for each chain.
                - num_chains (int): The total number of chains.
                - chain_id_dict (dict): A dictionary containing chain IDs for each residue.

        """
        chain_id_dict = {}  # Dictionary to store chain IDs for each residue
        chain_ids = []
        chain_starts = []  # Record the starting residue number for each chain

        for model in self.structure:
            for chain in model:
                chain_id = chain.get_id()
                chain_ids.append(chain_id)
                chain_index = len(chain_ids) - 1

                # Get the starting residue for the chain
                first_residue = next(chain.get_residues(), None)
                if first_residue:
                    chain_starts.append(first_residue.get_id()[1]-1)  # Get the residue number - 1

                # Iterate over all residues in the chain to store chain ID for each residue
                for residue in chain:
                    chain_id_dict[residue.get_id()] = chain_index    

        return chain_ids, chain_starts, len(chain_ids), chain_id_dict


    def __add_global_node(self, x, pos, edge_index, edge_attr=None, annotations=None):
        # Compute global node features
        global_node_feature = torch.cat([self.cls_embedding, x[:, 320:].mean(dim=0, keepdim=True)], dim=1)
        x = torch.cat([x, global_node_feature], dim=0)
        
        # Compute global node coordinates
        global_node_pos = pos.mean(dim=0, keepdim=True)
        pos = torch.cat([pos, global_node_pos], dim=0)
        
        # Update edge list to connect global node and all other nodes
        num_nodes = x.size(0)
        global_node_index = num_nodes - 1
        new_edges = torch.tensor([[i, global_node_index] for i in range(global_node_index)] +
                                [[global_node_index, i] for i in range(global_node_index)], dtype=torch.long).t()
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        
        # if edge_attr is not None, create edge attributes for new edges (e.g., zeros or a specific value)
        if edge_attr is not None:
            global_edge_attr = torch.zeros((new_edges.size(1), edge_attr.size(1)))

            edge_attr = torch.cat([edge_attr, global_edge_attr], dim=0)
        
        # if there are annotations, pad them with zeros
        if annotations is not None:
            global_annotations = torch.zeros((1, annotations.size(1)))
            annotations = torch.cat([annotations, global_annotations], dim=0)
        
        return x, pos, edge_index, edge_attr, annotations


    # ==============================
    # Main Methods
    # ==============================
    def __call__(self, file_path, Hetero=False, ESM=False, seq=True, loop=False, 
                            k=-1, d_cutoff=8.0, y=1, **kwargs):

        """
        Builds a protein graph from a PDB file.

        Args:
            file_path (str): The path to the PDB file.
            Hetero (bool, optional): Whether to create a HeteroData object. Defaults to False.
            ESM (bool, optional): Whether to use ESM embeddings. Defaults to False.
            seq (bool, optional): Whether to use sequence information. Defaults to True.
            loop (bool, optional): Whether to include loop edges in the graph. Defaults to False.
            k (int, optional): The number of nearest neighbors to consider. Defaults to -1.
            d_cutoff (float, optional): The distance cutoff for neighborhood extraction. Defaults to 8.0.
            y (int, optional): The label for the protein graph. Defaults to 1.
            **kwargs: Additional keyword arguments for feature extraction.

        Returns:
            protein_graph (HeteroData/Data): The protein graph containing node features, positions, and edge information.
        """

        # Parse the PDB file
        self.__parse_pdb(file_path)
        features = self.__feature_extraction(ESM, **kwargs)
        
        # Get node features
        x = features.get('residue_features', None)

        pos = torch.FloatTensor([list(coord) for coord in self.coordinates.values()])
        # x = torch.cat([pos, x], dim=1) # 3

        # Create edge index
        edge_index_seq, edge_index_dist, edge_index_knn= self.__extract_neighborhood(d_cutoff, k, seq)

        if self.pfam_df is not None:
            pfam_fragments = self.__extract_and_annotate_pfam_fragments().float()

        types = ['SEQ', 'DIST', 'KNN']    
        # Create a HeteroData object if Hetero is True
        if Hetero:
            protein_graph = HeteroData()
            protein_graph['protein'].x = x.float()
            protein_graph['protein'].pos = pos
            if self.pfam_df is not None:
                protein_graph['protein'].annotations = pfam_fragments
            for type in types:
                edge_index = edge_index_seq if type == 'SEQ' else edge_index_dist if type == 'DIST' else edge_index_knn
                if edge_index is not None:
                    protein_graph['protein', type, 'protein'].edge_index = edge_index
                    protein_graph['protein', type, 'protein'].edge_attr = relative_geometry_encoding(features.get('atom_positions'), edge_index).float()
        # Create a Data object if Hetero is False
        else:
            edge_indices = [e for e in [edge_index_seq, edge_index_dist, edge_index_knn] if e is not None]
            combined_edge_index = torch.cat(edge_indices, dim=1) 
            edge_index = torch.unique(combined_edge_index, dim=1)
            if loop:
                # Add self-loops after combining and deduplicating edges
                edge_index, _ = add_self_loops(edge_index)
            edge_attr = relative_geometry_encoding(features.get('atom_positions'), edge_index).float()
            if kwargs.get('add_edge', False):
                x = self.__add_edge_attr(x=x, edge_index=edge_index, edge_attr=edge_attr, esm_use=ESM)
            # x, pos, edge_index, edge_attr, pfam_fragments = self.__add_global_node(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, annotations=pfam_fragments)
            protein_graph = Data(x=x.float(), pos=pos, edge_index=edge_index, edge_attr=edge_attr)
            # protein_graph.dist_edge_index = self.__get_dist_edge_index(edge_index, protein_graph.num_nodes)
            # if self.update == True:
            #     protein_graph = self.__update_x(protein_graph)
        
        if self.pfam_df is not None:
            protein_graph.annotations = pfam_fragments
        if self.labels:
            protein_graph.y = self.__add_labels()
        else:
            if not isinstance(y, (int, float, torch.Tensor)):
                raise ValueError("The value of 'y' must be a number or a tensor")
            protein_graph.y = y

        protein_graph.prot_idx = self.prot_idx
        protein_graph.sequence = self.sequence
        if self.molecule_features:
            protein_graph.molecule_features = features['molecule_features']
        if self.fuz:
            protein_graph.llps = self.llps_value
        
        return protein_graph
    
def has_self_loops(edge_index):
    return torch.any(edge_index[0] == edge_index[1])


if __name__ == "__main__":
    builder = PDB2Graph(
        # pfam_path="data/uniprotkb/interproscan_pfam_output.tsv",
        # label_path="data/rg4bp/rg4bp_binding_vectors.csv",
        model_name = 'esm2_t33_650M_UR50D',
        layer_num = 33,
        hub_dir = 'code/esm2_t33_650M_UR50D',
                        )
    y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unsqueeze(0)
    protein_graph = builder("data/TrEMBL/AF-A0A7S5C2Z3-F1-model_v4.pdb", 
                            ESM=True, 
                            # fuz=True,
                            # add_edge=True, 
                            loop=True, 
                            k=10, d_cutoff=10.0,
                            y=y,
                            )
    
    print(protein_graph)