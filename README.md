# G4SAINT (Anonymous Submission)

This repository provides the code for:

**G4SAINT Enables Interpretable Prediction of RNA G-Quadruplex Binding Proteins across the Human Proteome via Domain-Guided Learning**

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
cd code
python datasets.py
```

This step generates graph representations and ESM2-based features required for training and evaluation.

Note: Preprocessed datasets are not included due to size but can be regenerated.

---

## Training

```bash
cd code
python train.py
```

---

## Evaluation

```bash
cd code
python test.py
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
cd code
python filter.py
```

---

## Pretrained Models

Pretrained weights are available at:

* G4SAINT model:
  https://zenodo.org/records/19845817/files/trial0015_d__epoch=22_val_loss=0.132.ckpt?download=1

* SAE model:
  https://zenodo.org/records/19845817/files/sae-epoch=02-train_loss=0.40.ckpt?download=1

After downloading, place them under:

```
code/model/new_train/
code/model/sae/
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