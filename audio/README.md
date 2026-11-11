# Audio recordings

This directory contains the 90 WAV recordings used in the two-session corpus for the associated paper:

**A Playing-Intensity-Normalized Multi-Feature Acoustic Representation for Comparing Plucking Positions in Tupipa Heritage Audio**

## Structure

```text
audio/
├── session1/  # 45 recordings
└── session2/  # 45 recordings
```

Each filename follows:

```text
session{session}_{position}_{intensity}[_{take}].wav
```

The filename without a suffix denotes take 0. Suffixes `_1` through `_4` denote takes 1 through 4. The position labels are measured from the bridge along the analyzed open string:

- Position 1: 3.0 cm, near the fushou and bridge.
- Position 2: 14.5 cm, the middle reference position.
- Position 3: 24.5 cm, near the fret region.

P, MF, and F are nominal performance-protocol labels. They are not physical force measurements.

## Authorized use

The recordings are publicly accessible for non-commercial research, teaching, archival study, and reproduction of the associated paper. Cite the paper when using the recordings or derived features. Do not sell, sublicense, or repost copies in another public repository or dataset; do not use the recordings for commercial model training, product development, or commercial sound libraries; and do not remove the cultural and recording-context information. Contact the corresponding author identified in the paper before any other use.
