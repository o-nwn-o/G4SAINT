import os
import csv
import json
import random
from sklearn.model_selection import KFold

import utils 

os.chdir(os.path.expanduser("~"))

class ClusterDataManager:
    def __init__(self, output_dir, seed=None):
        self.clstr_path = os.path.join(output_dir, "cluster.fasta.clstr")
        self.output_dir = output_dir
        self.seed = seed
        self.cluster_dict = {}  # Dict[str, List[str]]
        self.test_cluster_ids = []
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(os.path.join(output_dir, "clusters.json")):
            self.parse_clstr()
        else:
            with open(os.path.join(output_dir, "clusters.json"), "r") as f:
                self.cluster_dict = json.load(f)

    def parse_clstr(self):
        """Read clusters from CD-HIT .clstr file"""
        clusters = []  # List[List[str]]
        current_cluster = []
        with open(self.clstr_path, "r") as f:
            for line in f:
                if line.startswith(">Cluster"):
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = []
                else:
                    seq_id = line.split(">")[1].split("...")[0].split('|')[1]
                    current_cluster.append(seq_id)
        if current_cluster:
            clusters.append(current_cluster)

        # Save as a dictionary with numbered keys
        self.cluster_dict = {f"cluster_{i}": cluster for i, cluster in enumerate(clusters)}

        # Save as JSON
        with open(os.path.join(self.output_dir, "clusters.json"), "w") as f:
            json.dump(self.cluster_dict, f, indent=2)

    def split(self, train_ratio=0.8, valid_ratio=0.1):
        """Split clusters into train/valid/test and save ID lists as txt"""
        cluster_ids = list(self.cluster_dict.keys())
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(cluster_ids)

        total = len(cluster_ids)
        train_end = int(train_ratio * total)
        valid_end = train_end + int(valid_ratio * total)

        train_ids = cluster_ids[:train_end]
        valid_ids = cluster_ids[train_end:valid_end]
        test_ids = cluster_ids[valid_end:]

        self.test_cluster_ids = test_ids

        # Save cluster IDs
        with open(os.path.join(self.output_dir, "train_cluster_ids.json"), "w") as f:
            json.dump(train_ids, f, indent=2)

        with open(os.path.join(self.output_dir, "valid_cluster_ids.json"), "w") as f:
            json.dump(valid_ids, f, indent=2)
        
        with open(os.path.join(self.output_dir, "test_cluster_ids.json"), "w") as f:
            json.dump(test_ids, f, indent=2)

        # Flatten IDs and write to files
        self._write_id_list(train_ids, "train.txt")
        self._write_id_list(valid_ids, "valid.txt")
        self._write_id_list(test_ids, "test.txt")

    def _write_id_list(self, cluster_ids, filename):
        ids = [seq for cid in cluster_ids for seq in self.cluster_dict[cid]]
        print(f"Number of clusters in {filename}: {len(cluster_ids)}")
        print(f"Writing {len(ids)} IDs to {filename}")
        with open(os.path.join(self.output_dir, filename), "w") as f:
            f.write("\n".join(ids) + "\n")


    # ==============================
    # Dataset processing
    # ==============================
    def resample_train_set(self, class_name, pos_to_neg_ratio=1):
        """
        Resample the training set IDs and binding vectors to balance a specified class.

        Args:
            class_name (str): Target class name, e.g., 'all_RNA_G4_Binding'.
            output_path (str): Path to save the resampled IDs.
            pos_to_neg_ratio (int): Positive-to-negative sample ratio, default is 1:1.
        """
        positive_samples, negative_samples = self.get_ids_for_class(class_name)

        num_pos = len(positive_samples)
        num_neg_sampled = min(len(negative_samples), num_pos * pos_to_neg_ratio)
        if self.seed is not None:
            random.seed(self.seed)
        sampled_negatives = random.sample(negative_samples, num_neg_sampled)

        final_ids = sorted(set(positive_samples + sampled_negatives))

        output_path = class_name + '_' + str(pos_to_neg_ratio) + ".txt"
        # Save
        with open(os.path.join(self.output_dir, output_path), "w") as f:
            f.write("\n".join(final_ids) + "\n")

        print(f"[Resample Summary]")
        print(f"  ➤ Total sampled: {len(final_ids)}")
        print(f"  ➤ Positives: {len(positive_samples)}")
        print(f"  ➤ Negatives (sampled): {len(sampled_negatives)}")

        return os.path.join(self.output_dir, output_path)
    

    def get_ids_for_class(self, class_name):
        """
        Only extract positive sample IDs for a given class.
        """
        # Read IDs
        uniprot_ids = utils.read_uniprot_ids([os.path.join(self.output_dir, "train.txt")])
        train_ids = uniprot_ids.get("train", set())

        # Load class index mapping
        with open("data/uniprotkb_index_mapping.json", 'r') as f:
            index_mapping = json.load(f)
        class_index = index_mapping[class_name]

        # Read binding vectors
        binding_data = {}
        with open("data/uniprotkb_binding_vectors.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                uid = row[0]
                binding_vector = json.loads(row[1])
                binding_data[uid] = binding_vector

        # Class-based positive sample extraction
        positive_samples = []
        negative_samples = []
        for uid in train_ids:
            if uid in binding_data:
                label = binding_data[uid][class_index]
                if label == 1:
                    positive_samples.append(uid)
                elif label == 0:
                    negative_samples.append(uid)

        return positive_samples, negative_samples


    def generate_cross_validation(self, k=5):
        """Perform k-fold cross-validation on non-test clusters and return all fold splits"""
        with open(os.path.join(self.output_dir, "clusters.json"), "r") as f:
            cluster_dict = json.load(f)
        with open(os.path.join(self.output_dir, "test_cluster_ids.json"), "r") as f:
            test_ids = set(json.load(f))

        remaining_clusters = {
            cid: seqs for cid, seqs in cluster_dict.items() if cid not in test_ids
        }
        cluster_keys = list(remaining_clusters.keys())

        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(cluster_keys)

        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        self.folds = []

        for fold_index, (train_idx, valid_idx) in enumerate(kf.split(cluster_keys)):
            train_clusters = [remaining_clusters[cluster_keys[i]] for i in train_idx]
            valid_clusters = [remaining_clusters[cluster_keys[i]] for i in valid_idx]

            train_flat = [seq for cluster in train_clusters for seq in cluster]
            valid_flat = [seq for cluster in valid_clusters for seq in cluster]

            self.folds.append({
                "fold": fold_index,
                "train_seqs": train_flat,
                "valid_seqs": valid_flat
            })

        return self.folds
    
    def load_cross_validation_split(self, fold_id, fold_splits, k=5):
        """
        fold_id: Current fold index
        fold_splits: List[Dict[str, List[str]]], the return value of generate_cross_validation()
        Returns: train_dataset, valid_dataset
        """
        if not hasattr(self, 'folds'):
            fold_splits = self.generate_cross_validation(k)
        if fold_id >= len(fold_splits):
            raise ValueError(f"Invalid fold_id: {fold_id}. Must be less than {len(fold_splits)}.")
        if fold_id < 0:
            raise ValueError(f"Invalid fold_id: {fold_id}. Must be non-negative.")
        
        ids_dict = {
            "train": set(fold_splits[fold_id]["train_seqs"]),
            "valid": set(fold_splits[fold_id]["valid_seqs"]),
        }

        return ids_dict
    

    def extract_representatives(self, split="test", output_filename=None):
        """
        Extract one representative sequence (default: first or random) from each cluster in the specified set (train/valid/test),
        ensuring low similarity between sequences.

        Args:
            split (str): "train", "valid", "test"
            output_filename (str): Output filename, default is "{split}_cluster_representatives.txt"
        Returns:
            str: Path to the saved file
        """
        assert split in ("train", "valid", "test"), "split must be 'train', 'valid', or 'test'"
        if output_filename is None:
            output_filename = f"{split}_cluster_representatives.txt"

        # Read clusters.json
        with open(os.path.join(self.output_dir, "clusters.json"), "r") as f:
            cluster_dict = json.load(f)

        # Read cluster IDs for the specified split
        with open(os.path.join(self.output_dir, f"{split}_cluster_ids.json"), "r") as f:
            cluster_ids = json.load(f)

        representative_ids = []

        # Set random seed
        if self.seed is not None:
            random.seed(self.seed)

        for cid in cluster_ids:
            seqs = cluster_dict[cid]
            if not seqs:
                continue
            # Randomly select one representative sequence (or always pick the first)
            rep_seq = random.choice(seqs)
            representative_ids.append(rep_seq)

        # Save
        save_path = os.path.join(self.output_dir, output_filename)
        with open(save_path, "w") as f:
            f.write("\n".join(representative_ids) + "\n")

        print(f"[{split.capitalize()} Representatives Summary]")
        print(f"  ➤ {split.capitalize()} clusters: {len(cluster_ids)}")
        print(f"  ➤ Representatives saved: {len(representative_ids)}")
        print(f"  ➤ Saved to: {save_path}")

        return save_path


if __name__ == "__main__":
    manager = ClusterDataManager(output_dir="data/uniprotkb/")
    # manager.split(train_ratio=0.6, valid_ratio=0.2)
    # name = manager.resample_train_set('all_RNA_G4_Binding', 2)
    name = manager.extract_representatives()
    print(name)