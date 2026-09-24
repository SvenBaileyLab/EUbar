# EUbar pre-refactor regression tests

This is intentionally a **small behavior-freezing suite**, added before the YAML/config refactor.
It is not the final comprehensive test suite.

Covered now:

- Holm adjustment and AFF/RAND family separation
- `--best-pval` RAND-support selection semantics
- fast contiguous SNV matching vs the original full matcher
- arbitrary/asymmetric masks and reverse-complement signature handling
- deterministic RAND sampling
- max-probes strict recommendation + borderline reporting
- matched random-mask controls and control-exclusion semantics
- masked motif seed representation, spacer information content, and PPM reverse complement

Run from the repository root:

```bash
PYTHONPATH=. pytest -q
```

The later comprehensive suite should add end-to-end CLI fixtures for array, intensities, scan,
SNV (contiguous + masked + multiprocessing), motifs (standard + masked), calibrations, YAML,
and project/pipeline execution.
