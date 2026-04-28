# residue_properties

# 1
# Figure S6 A
# Nucleic Acid Binding - mRNAinteractome, Castello et al., Cell 2011, 149:1393-1406
NAB_mRNAinteractome = {
    'A': -0.10, 'R': 0.33, 'N': 0.04,  'D': -0.00, 'C': -0.60,
    'Q': -0.03, 'E': -0.00, 'G': 0.20,  'H': -0.11, 'I': -0.12,
    'L': -0.23, 'K': 0.37,  'M': -0.41, 'F': -0.05, 'P': -0.03,
    'S': -0.07, 'T': -0.11, 'W': -0.18, 'Y': 0.21,  'V': -0.10
}
# 3
# Nucleic Acid Binding - classicalRBD, Castello et al., Cell 2011, 149:1393-1406
NAB_classicalRBD = {
'A': -0.23, 'R': 0.47,  'N': 0.12, 'D': 0.02,  'C': -0.94,
'Q': -0.05, 'E': -0.25,  'G': 0.51, 'H': -0.13, 'I': -0.21,
'L': -0.47, 'K': -0.00, 'M': -0.56, 'F': -0.01,  'P': 0.15,
'S': 0.16, 'T': -0.24, 'W': -0.22,  'Y': 0.64, 'V': -0.24
}
# Nucleic Acid Binding - nonclassicalRBD, Castello et al., Cell 2011, 149:1393-1406
NAB_nonclassicalRBD = {
    'A': -0.04,  'R': 0.42,  'N': -0.10, 'D': -0.20, 'C': -0.57,
    'Q': -0.05,  'E': -0.09,  'G': 0.20, 'H': -0.07, 'I': -0.03,
    'L': -0.19, 'K': 0.61,  'M': -0.29, 'F': 0.02, 'P': -0.11,
    'S': -0.36, 'T': -0.09,  'W': 0.02,  'Y': 0.01, 'V': 0.04
}
# 6
# Nucleic Acid Binding - unknownRBD, Castello et al., Cell 2011, 149:1393-1406
NAB_unknownRBD = {
    'A': -0.05, 'R': 0.18,  'N': -0.10, 'D': 0.08, 'C': -0.45,
    'Q': -0.01, 'E': 0.16, 'G': -0.00, 'H': -0.12, 'I': -0.13,
    'L': -0.13, 'K': 0.42,  'M': -0.41, 'F': -0.12, 'P': -0.10,
    'S': -0.07, 'T': -0.06, 'W': -0.27,  'Y': 0.02,  'V': -0.11
}


# Normalized frequency of alpha-helix in all-alpha class, Palau et al., Int. J. Peptide Protein Res. 1981, 19:394-401
# https://www.genome.jp/dbget-bin/www_bget?aaindex:PALJ810115
alpha_helix_Palau = {
    'A': 0.91, 'R': 0.77, 'N': 1.32, 'D': 0.90, 'C': 0.50,
    'Q': 1.06, 'E': 0.53, 'G': 1.61, 'H': 1.08, 'I': 0.36,
    'L': 0.77, 'K': 1.27, 'M': 0.76, 'F': 0.37, 'P': 1.62,
    'S': 1.34, 'T': 0.87, 'W': 1.10, 'Y': 1.24, 'V': 0.52
}


# 7
# Energy transfer from out to in(95%buried), Radzicka-Wolfenden, Biochemistry 1988, 27:1664-1670
# https://www.genome.jp/entry/aaindex:RADA880107
Burial_Radzicka = {
'A': -0.29, 'R': -2.71, 'N': -1.18, 'D': -1.02, 'C': 0.0,
'Q': -1.53, 'E': -0.90, 'G': -0.34, 'H': -0.94, 'I': 0.24,
'L': -0.12, 'K': -2.05, 'M': -0.24, 'F': 0.0, 'P': 0.0,
'S': -0.75, 'T': -0.71, 'W': -0.59, 'Y': -1.02, 'V': 0.09
}

# 10
# Percentage of buried residues, Janin et al., J. Mol. Biol. 1978, 125:357-386
# https://www.genome.jp/entry/aaindex:JANJ780102
Burial_Janin = {
    'A': 51.0, 'R': 5.0, 'N': 22.0, 'D': 19.0, 'C': 74.0,
    'Q': 16.0, 'E': 16.0, 'G': 52.0, 'H': 34.0, 'I': 66.0,
    'L': 60.0, 'K': 3.0, 'M': 52.0, 'F': 58.0, 'P': 25.0,
    'S': 35.0, 'T': 30.0, 'W': 49.0, 'Y': 24.0, 'V': 64.0
}

# Proportion of residues 100% buried, Chothia, J. Mol. Biol. 1976, 105:1-14
# https://www.genome.jp/entry/aaindex:CHOC760104
Burial_Chothia = {
    'A': 0.20, 'R': 0.00, 'N': 0.03, 'D': 0.04, 'C': 0.22,
    'Q': 0.01, 'E': 0.03, 'G': 0.18, 'H': 0.02, 'I': 0.19,
    'L': 0.16, 'K': 0.00, 'M': 0.11, 'F': 0.14, 'P': 0.04,
    'S': 0.08, 'T': 0.08, 'W': 0.04, 'Y': 0.03, 'V': 0.18
}


# Knowledge-based membrane-propensity scale from 1D_Helix in MPtopo databases, Punta-Maritan, Proteins 2003, 50:114-121
# https://www.genome.jp/entry/aaindex:PUNT030101
membrane_Punta1 = {
    'A': -0.17, 'R': 0.37, 'N': 0.18, 'D': 0.37, 'C': -0.06,
    'Q': 0.26, 'E': 0.15, 'G': 0.01, 'H': -0.02, 'I': -0.28,
    'L': -0.28, 'K': 0.32, 'M': -0.26, 'F': -0.41, 'P': 0.13,
    'S': 0.05, 'T': 0.02, 'W': -0.15, 'Y': -0.09, 'V': -0.17
}

# Knowledge-based membrane-propensity scale from 3D_Helix in MPtopo databases, Punta-Maritan, Proteins 2003, 50:114-121
# https://www.genome.jp/entry/aaindex:PUNT030102
membrane_Punta2 = {
    'A': -0.15, 'R': 0.32, 'N': 0.22, 'D': 0.41, 'C': -0.15,
    'Q': 0.03, 'E': 0.30, 'G': 0.08, 'H': 0.06, 'I': -0.29,
    'L': -0.36, 'K': 0.24, 'M': -0.19, 'F': -0.22, 'P': 0.15,
    'S': 0.16, 'T': -0.08, 'W': -0.28, 'Y': -0.03, 'V': -0.24
}


# Hydrophobicity, Eisenberg et al, J. Mol. Biol. 1984, 179:125-142
# https://web.expasy.org/protscale/pscale/Hphob.Eisenberg.html
hydrophobicity_Eisenberg = {
    'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'Q': -0.850, 'E': -0.740, 'G': 0.480, 'H': -0.400, 'I': 1.380,
    'L': 1.060, 'K': -1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.080
}

# 9
# Hydrophobicity, Janin, Nature 1979, 277:491-492
# https://web.expasy.org/protscale/pscale/Hphob.Janin.html
hydrophobicity_Janin = {
    'A': 0.300, 'R': -1.400, 'N': -0.500, 'D': -0.600, 'C': 0.900,
    'Q': -0.700, 'E': -0.700, 'G': 0.300, 'H': -0.100, 'I': 0.700,
    'L': 0.500, 'K': -1.800, 'M': 0.400, 'F': 0.500, 'P': -0.300,
    'S': -0.100, 'T': -0.200, 'W': 0.300, 'Y': -0.400, 'V': 0.600
}

# 5
# Hydrophobicity, Fauchere et al., Eur. J. Med. Chem. 1983, 18:369-375
# https://web.expasy.org/protscale/pscale/Hphob.Fauchere.html
hydrophobicity_Fauchere = {
    'A': 0.310, 'R': -1.010, 'N': -0.600, 'D': -0.770, 'C': 1.540, 
    'Q': -0.220, 'E': -0.640, 'G': 0.000, 'H': 0.130, 'I': 1.800, 
    'L': 1.700, 'K': -0.990, 'M': 1.230, 'F': 1.790, 'P': 0.720, 
    'S': -0.040, 'T': 0.260, 'W': 2.250, 'Y': 0.960, 'V': 1.220
}

# 2
# Hydrophobicity, Abraham & Leo, Proteins: Structure; Function and Genetics 1987, 2:130-152
# https://web.expasy.org/protscale/pscale/Hphob.Leo.html
hydrophobicity_Leo = {
    'A': 0.440, 'R': -2.420, 'N': -1.320, 'D': -0.310,
    'C': 0.580, 'Q': -0.710, 'E': -0.340, 'G': 0.000,
    'H': -0.010, 'I': 2.460, 'L': 2.460, 'K': -2.450,
    'M': 1.100, 'F': 2.540, 'P': 1.290, 'S': -0.840,
    'T': -0.410, 'W': 2.560, 'Y': 1.630, 'V': 1.730
}

residue_properties = {
    'A': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 89.09, 'volume': 88.6},
    'R': {'polarity': 1, 'charge': 1, 'hydrophobic': 0, 'aromatic': 0, 'mass': 174.2, 'volume': 202},
    'N': {'polarity': 1, 'charge': 0, 'hydrophobic': 0, 'aromatic': 0, 'mass': 132.12, 'volume': 114.1},
    'D': {'polarity': 1, 'charge': -1, 'hydrophobic': 0, 'aromatic': 0, 'mass': 133.1, 'volume': 111.1},
    'C': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 121.15, 'volume': 108.5},
    'Q': {'polarity': 1, 'charge': 0, 'hydrophobic': 0, 'aromatic': 0, 'mass': 146.15, 'volume': 143.8},
    'E': {'polarity': 1, 'charge': -1, 'hydrophobic': 0, 'aromatic': 0, 'mass': 147.13, 'volume': 138.4},
    'G': {'polarity': 0, 'charge': 0, 'hydrophobic': 0, 'aromatic': 0, 'mass': 75.07, 'volume': 60.1},
    'H': {'polarity': 1, 'charge': 1, 'hydrophobic': 0, 'aromatic': 1, 'mass': 155.15, 'volume': 153.2},
    'I': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 131.17, 'volume': 166.7},
    'L': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 131.17, 'volume': 166.7},
    'K': {'polarity': 1, 'charge': 1, 'hydrophobic': 0, 'aromatic': 0, 'mass': 146.19, 'volume': 168.6},
    'M': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 149.21, 'volume': 162.9},
    'F': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 1, 'mass': 165.19, 'volume': 189.9},
    'P': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 115.13, 'volume': 112.7},
    'S': {'polarity': 1, 'charge': 0, 'hydrophobic': 0, 'aromatic': 0, 'mass': 105.09, 'volume': 89},
    'T': {'polarity': 1, 'charge': 0, 'hydrophobic': 0, 'aromatic': 0, 'mass': 119.12, 'volume': 116.1},
    'W': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 1, 'mass': 204.23, 'volume': 227.8},
    'Y': {'polarity': 1, 'charge': 0, 'hydrophobic': 0, 'aromatic': 1, 'mass': 181.19, 'volume': 193.6},
    'V': {'polarity': 0, 'charge': 0, 'hydrophobic': 1, 'aromatic': 0, 'mass': 117.15, 'volume': 140},
}

