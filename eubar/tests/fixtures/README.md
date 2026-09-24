These fixtures were recorded from commit e47fda5c2550ccb1ac8ccbde49eb2d514e1dd747.
The reference checkout was kept separate from the refactor.

The synthetic genome and 700 probes use Python random seed 2718. Probe lengths
vary from 18 to 30 bases; the BED repeats its first interval. Intensities contain
Gaussian noise and an ACGT/ATGT effect. Three variants exercise actual fits,
including missing support. This is a compatibility fixture, not biological data.

`execution_cases.py` describes the reference commands. Expected TSV/MEME files
and SNV/scan stdout are compared byte for byte. PNGs are checked for successful
rendering, not byte equality. `reference-environment.txt` records the environment
used to capture the numerical results; different numerical-library versions may
change final floating-point digits. Such differences require investigation, not
automatic replacement of these files.

Intensity extraction also requires bedtools (reference: 2.31.0).
