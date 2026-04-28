# G4SAINT (Anonymous Submission)

This repository provides the code for:

**G4SAINT Enables Interpretable Prediction of RNA G-Quadruplex Binding Proteins across the Human Proteome via Domain-Guided Learning**

---

## Installation

```bash
pip install -r requirements.txt
````

---

## Data Preparation

### 1. Protein sequences (FASTA)

Prepare FASTA file for proteins.

### 2. InterProScan

Run:

```bash
interproscan.sh -i input.fasta -f tsv -appl Pfam -o interpro.tsv
```

Place output:

```
data/TrEMBL/interproscan_pfam_output.tsv
```

---

### 3. Protein structures

Download AlphaFold PDB files and place in:

```
data/TrEMBL/
```

---

## Pretrained Models

Pretrained weights are available at:

- G4SAINT model:

  https://zenodo.org/records/19845817/files/trial0015_d__epoch=22_val_loss=0.132.ckpt?download=1
  

- SAE model:
  https://zenodo.org/records/19845817/files/sae-epoch=02-train_loss=0.40.ckpt?download=1

After downloading, place them under:

```
code/model/new_train/
code/model/sae/
```

---

## Run Screening

```bash
cd code
python filter.py
```

---

## Test

```bash
cd code
python test.py
```

---

## Notes

### ESM2 Protein Language Model

We use the ESM2 model:

- Model: esm2_t33_650M_UR50D
- Source: https://github.com/facebookresearch/esm
- Version: ESM2 (Rives et al., 2021)

The model weights can be automatically downloaded via the ESM package, 
or manually obtained from the official repository.

If downloaded manually, place the weights under:

```
code/esm2_t33_650M_UR50D/
```

