# EUbar tests

From the repository root:

```bash
python -m pip install -e '.[test]'
PYTHONPATH=. pytest -q
```

Install the external `bedtools` executable for the intensity extraction test.
That test reports a skip if the executable is absent; the reference verification
included it. Plot rendering uses Matplotlib's Agg backend.

The original 27 tests remain. The additional suite covers:

- Exact array output, duplicate intervals, ambiguous bases and legacy flags.
- Default GC residualization output from an actual BigWig.
- Real contiguous, symmetric-mask and asymmetric-mask SNV fits, RAND,
  best-p-value selection, Holm, probe caps and diagnostic columns.
- Repeated serial/fork equality and separate spawn-worker checks.
- Region scan, pooled scan, Holm scan and reverse-strand output.
- Ordinary, extended and masked motifs: PPM, reduced tables and MEME output,
  with PNG rendering checks.
- Both calibrations, matched random controls, recommendation tables and PDF
  rendering checks.
- Direct YAML/API equivalence, paths, defaults, overrides, pipelines, cycle
  detection, failure propagation and preserved public help.
- Missing tests, tied effects, unsupported RAND fallback, empty results and
  p-value underflow formatting.

`fixtures/expected` was captured from the unmodified e47fda5 commit. The test
commands are in `execution_cases.py`. Tables require exact headers, row order, identifiers, counts, discrete decisions
and missing values. Listed measured columns allow relative differences up to
1e-12, with no absolute tolerance (including for tiny p-values). This permits
last-digit rounding differences between machines without replacing the baseline.
Array and MEME files, CLI help, and same-machine serial/fork output remain exact.
PNG/PDF bytes are not suitable as scientific goldens.
The reference dependency versions are in `fixtures/reference-environment.txt`.
Investigate numerical-library differences before updating expected files.

Synthetic inputs deliberately include sparse and rank-deficient cases. The
known rank-deficiency warnings in masked YAML cases and invalid-log1p warnings
in the pooled YAML case are filtered by message, category and source module
only in those cases. Other warnings and real analysis warnings remain visible. The suite is a compatibility check, not a replacement for validating
the user's full biological datasets.
