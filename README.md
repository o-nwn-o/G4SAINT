# G4SAINT

This repository provides the source code for:

**G4SAINT Enables Interpretable Prediction of RNA G-Quadruplex Binding Proteins across the Human Proteome via Domain-Guided Learning**

## Code and Model Availability

The source code for G4SAINT is publicly available at:

- GitHub: https://github.com/o-nwn-o/G4SAINT

The pretrained G4SAINT and sparse autoencoder (SAE) model weights are permanently archived on Zenodo:

- Zenodo record: https://zenodo.org/records/23555760

The repository provides instructions for obtaining the source datasets and reproducing data preprocessing, model training, evaluation, and proteome-wide screening.

---

## Environment Setup

The code has been tested in the following environment:

- Python: 3.11  
- PyTorch: 2.5  
- CUDA (PyTorch): 12.1  
- OS: Linux  

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training and Evaluation Data

The model is trained and evaluated on the **human protein subset from the January 2025 release of UniProt Knowledgebase (UniProtKB)**.

Minimal dataset files are provided under:

```
data/
```

See `data/README.md` for detailed descriptions of each file.

### Preparing structural data

Download AlphaFold PDB structures for proteins listed in:

```
data/uniprotkb_ids.txt
```

and place them under:

```
data/uniprotkb/
```

### Generating the dataset

Run:

```bash
python code/datasets.py
```

This step generates graph representations and ESM2-based features required for training and evaluation.

Note: Preprocessed datasets are not included due to size but can be regenerated.

---

## Training

```bash
python code/train.py
```

---

## Evaluation

```bash
python code/test.py
```

---

## Proteome-wide Screening (Application)

### Required inputs

For applying the model to new proteins, the following inputs are required:

* Protein FASTA
* InterProScan annotation
* AlphaFold PDB structures

### InterProScan

```bash
interproscan.sh -i input.fasta -f tsv -appl Pfam -o interpro.tsv
```

### Run screening

```bash
python code/filter.py
```

---

## Pretrained Models

The pretrained model weights used in this study are archived on Zenodo:

https://doi.org/10.5281/zenodo.23555760

| Model | Zenodo filename | Local destination |
|---|---|---|
| G4SAINT prediction model | `g4saint_predictor.ckpt` | `code/model/g4saint/g4saint_predictor.ckpt` |
| Sparse autoencoder (SAE) | `g4saint_sae.ckpt` | `code/model/sae/g4saint_sae.ckpt` |

Create the required directories:

```bash
mkdir -p code/model/g4saint code/model/sae
```

Download the pretrained G4SAINT model checkpoint:

```bash
curl -fL \
  "https://zenodo.org/records/23555760/files/g4saint_predictor.ckpt?download=1" \
  -o "code/model/g4saint/g4saint_predictor.ckpt"
```

Download the pretrained SAE model checkpoint:

```bash
curl -fL \
  "https://zenodo.org/records/23555760/files/g4saint_sae.ckpt?download=1" \
  -o "code/model/sae/g4saint_sae.ckpt"
```

The expected directory structure is:

```bash
code/model/
├── g4saint/
│   └── g4saint_predictor.ckpt
└── sae/
    └── g4saint_sae.ckpt
```

---

## Notes

### ESM2 Protein Language Model

We use the ESM2 model:

* Model: esm2_t33_650M_UR50D
* Source: https://github.com/facebookresearch/esm
* Version: ESM2 (Rives et al., 2021)

The model weights can be automatically downloaded or manually placed in:

```
code/esm2_t33_650M_UR50D/
```

---

### InterProScan

We used InterProScan version:

* Version: interproscan-5.75-106.0

Download from:

https://www.ebi.ac.uk/interpro/interproscan.html

Example:

```bash
interproscan.sh -i input.fasta -f tsv -appl Pfam -o interpro.tsv
```

Note: Different versions may produce slightly different domain annotations.

### Hardware

All experiments were conducted on a machine with:

- GPU: NVIDIA GeForce RTX 4090 (24 GB memory)