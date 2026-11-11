# Tupipa Heritage Audio: Reproducible Acoustic Comparison

This repository accompanies the paper **A Playing-Intensity-Normalized Multi-Feature Acoustic Representation for Comparing Plucking Positions in Tupipa Heritage Audio**.

It contains the 90 WAV recordings used in the two-session study, the per-recording feature table used by the reported evaluation, and the two scripts needed to regenerate the features and evaluate the formal 16-dimensional representation. Fitted parameters and evaluation outputs are generated locally and are not committed to this minimal repository.

## Contents

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── audio/
│   ├── session1/  # 45 WAV files
│   └── session2/  # 45 WAV files
├── metadata/
│   └── per_file_features.csv
└── code/
    ├── analyze_tupipa_features.py
    └── evaluate_formal_16d.py
```

## Recordings

The corpus contains two sessions, three plucking positions, three nominal playing-intensity labels (P, MF, and F), and five takes per condition. The recordings are mono, 44.1 kHz, 24-bit PCM WAV files. The performer and recording operator was the paper's author.

Filenames follow this pattern:

```text
session{session}_{position}_{intensity}[_{take}].wav
```

The filename without a take suffix denotes take 0; suffixes `_1` through `_4` denote takes 1 through 4. Plucking positions are measured from the bridge along the analyzed open string:

| Position | Distance | Location |
|---|---:|---|
| 1 | 3.0 cm | Near the fushou and bridge |
| 2 | 14.5 cm | Middle reference position |
| 3 | 24.5 cm | Near the fret region |

P, MF, and F are protocol-defined performance labels, not force-sensor measurements.

## Reproduce the analysis

Python 3.10 or newer and NumPy are required. From the repository root, run:

```bash
python3 -m pip install -r requirements.txt
python3 code/analyze_tupipa_features.py audio outputs/per_file_features.csv
python3 code/evaluate_formal_16d.py outputs/per_file_features.csv outputs/reproduction
```

The repository includes `metadata/per_file_features.csv`, the 90-row feature table used as input to the reported evaluation. To rerun that evaluation directly, use:

```bash
python3 code/evaluate_formal_16d.py metadata/per_file_features.csv outputs/reproduction
```

To regenerate the table from the public WAV files and then rerun evaluation, use:

```bash
python3 code/analyze_tupipa_features.py audio outputs/per_file_features_recomputed.csv
python3 code/evaluate_formal_16d.py outputs/per_file_features_recomputed.csv outputs/recomputed_results
```

The evaluator applies the deterministic quality-control rule, estimates intensity-specific centers and fixed scales from session 1, and evaluates session 2 as the held-out query set. It writes the fitted parameters, per-query retrieval outcomes, and a result summary under the specified output directory.

The main analysis should report 89 valid recordings after quality control and direct cross-intensity position retrieval of 29/44 (0.659) for the all-group-centered formal 16-dimensional representation. The uncentered multi-feature baseline should report 27/44 (0.614). These values are reference checks for reproducing the paper's reported analysis.

## Access and use

The WAV files and code are made publicly accessible through this repository for non-commercial research, teaching, archival study, and reproduction of the associated paper. Please cite the paper when using the recordings, code, or derived features. Do not sell, sublicense, or repost copies in another public repository or dataset; do not use the recordings for commercial model training, product development, or commercial sound libraries; and do not redistribute copies outside this repository without prior written permission from the author. Retain the supplied cultural and recording-context information, and do not use these materials to make unsupported claims about performers, communities, or cultural authenticity.

No separate open-source or commercial-use license has been assigned. Public access through this repository is limited to the uses stated above; other uses require permission from the author.
