# Data Preparation

This project requires:

## 1. FASTA

Protein sequences.

## 2. InterProScan output

Generated using:

```bash
interproscan.sh -i input.fasta -f tsv -appl Pfam -o interpro.tsv
````

## 3. AlphaFold structures

Download PDB files and place under:

```
data/TrEMBL/
```