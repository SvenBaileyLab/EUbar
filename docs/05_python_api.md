# Python task API

CLI commands and YAML jobs use the same task configuration and execution functions.
The functions do not parse command-line arguments. They retain the CLI's output
files, stdout tables, row order and return codes.

```python
from eubar.snv import run_snv
from eubar.task_config import SnvConfig

if __name__ == "__main__":
    config = SnvConfig(
        intensities="TF.int",
        array="regions_8mer.txt",
        genome="hg38.fa",
        snv_list="chr5:1295113:C>T",
        best_pval=True,
        holm=True,
        jobs=4,
    )
    status = run_snv(config)
```

For multiprocessing, put the call inside `if __name__ == "__main__":` in a
script. This is required on platforms that start workers with spawn.

| Task module | Function | Configuration |
| --- | --- | --- |
| `eubar.array` | `run_array` | `ArrayConfig` |
| `eubar.intensities` | `run_intensities` | `IntensitiesConfig` |
| `eubar.snv` | `run_snv` | `SnvConfig` |
| `eubar.scan` | `run_scan` | `ScanConfig` |
| `eubar.motifs` | `run_motifs` | `MotifsConfig` |
| `eubar.calibrate` | `run_max_probe_calibration` | `CalibrationConfig` |
| `eubar.calibrate` | `run_mask_calibration` | `CalibrationConfig` |

Configuration classes live in `eubar.task_config`. Defaults are defined there;
CLI aliases remain in the task parsers. For intensities the Python field is
`genome`, while the CLI retains `--genome-fasta` and YAML retains
`inputs.genome_fasta`.

`run_*` validates options before execution. Invalid configuration raises
`TaskConfigError` (a `ValueError`). Data-reading and numerical exceptions retain
the existing task behavior. Calibration may also raise `ValueError` for an
invalid calibration request. The CLI translates configuration errors into usage
errors. Calls return an integer status; results remain in the task's established
files or stdout. Use `contextlib.redirect_stdout` to capture table output.

For values loaded from a dictionary, `SnvConfig.from_values(values)` and the
corresponding methods on other config classes perform scalar conversion.
Calling `validate()` checks requirements and option combinations without running
an analysis. Inputs are paths as strings; Python and CLI paths resolve from the
working directory.

## YAML

For complete job files and CLI examples, see [YAML jobs](07_yaml.md).

`eubar.config.load_yaml(path)` loads a document; `task_options(document)` resolves
it to a task configuration and an optional stdout path. YAML paths resolve from
the document's directory. Precedence is defaults, then YAML, then explicit CLI
overrides. Only supplied override flags are parsed; YAML execution itself does
not reconstruct or parse argv. `build_argv` remains available for dry-run display
and compatibility with existing callers.

Pipelines continue to reference independent YAML files. They execute in order,
stop at the first nonzero status, and reject cycles. Mask calibration remains a
diagnostic report: its recommendation is never automatically applied to a task.

## Masks and calibration from Python

For masked SNVs, set `mask="111100001111"`, provide `genome` and
`intensities`, and omit `array`. The same matching and output rules described in
[SNV analysis](02_snv.md) apply. `max_probes` and the numeric `seed` control
matched-probe subsampling; neither changes the RAND background sample size.

```python
from eubar.calibrate import run_mask_calibration
from eubar.task_config import CalibrationConfig

status = run_mask_calibration(CalibrationConfig(
    intensities="TF.int",
    genome="hg38.fa",
    masks="11111111,111100001111,11111000011111",
    stability_probes=100,
    n_families=500,
    repeats=50,
    random_controls=20,
    out_prefix="results/mask_calibration",
    plot=True,
))
```

For `run_max_probe_calibration`, provide `array`, `intensities`, `kmer_size`
and `out_prefix`; `caps` is a tuple of integers. The two named calibration
functions select their calibration target. Read the report before setting
analysis options; neither function applies a recommendation to another task.
See [Masks and calibration](06_masks_and_calibration.md) for the criteria.

## Internal modules

- `core.analyze`, `design`, `regression`, `rand`, `matching` and `sampling` retain
  the scientific analysis implementations.
- `core.snv_results` handles within-SNV correction and shared-position selection;
  `core.reporting` formats their output. Scan no longer imports the SNV CLI.
- `core.patterns` owns span and informative-position metadata. No-mask jobs still
  use the historical contiguous matcher; an all-ones explicit mask still uses
  the explicit-mask path.
- `core.motif_data`, `motif_search`, `motif_mask` and `motif_output` separate input,
  search, masked search and output responsibilities. `motifs.py` orchestrates the
  task and retains imports of its earlier helper functions for compatibility.
- `core.calibration_reporting` writes tables and plots. Calibration's numerical
  implementation remains in `core.calibration`.
