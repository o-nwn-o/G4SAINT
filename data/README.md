# Dataset Description

This folder contains metadata and dataset splits derived from the **human protein subset of the January 2025 release of UniProt Knowledgebase (UniProtKB)**.

## Directory structure

data/
├── uniprotkb_binding_vectors.csv
├── uniprotkb_ids.txt
└── uniprotkb/
    ├── train.txt
    ├── valid.txt
    ├── test.txt
    ├── uniprotkb_index_mapping.json
    └── interproscan_pfam_output.tsv

---

## File descriptions

### Top-level files

- **uniprotkb_binding_vectors.csv**  
  Multi-label annotations for proteins used in training and evaluation.

- **uniprotkb_ids.txt**  
  UniProt IDs for all human proteins in the dataset (January 2025 UniProtKB release).  
  This file defines the full protein universe used in this study.

---

### Subdirectory: uniprotkb/

- **train.txt / valid.txt / test.txt**  
  Predefined dataset splits.

- **uniprotkb_index_mapping.json**  
  Mapping between label names and indices used in the model.

- **interproscan_pfam_output.tsv**  
  Domain annotations generated using InterProScan (Pfam).

---

## Notes

- Protein structures (AlphaFold PDB files) are not included and should be downloaded separately.
- Preprocessed datasets (e.g., ESM2 embeddings and graph representations) are not included due to size and can be generated using `code/datasets.py`.