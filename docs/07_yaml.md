# YAML jobs and pipelines

YAML records inputs and parameters so you can rerun a task without rebuilding a
long command. Install YAML support with `python -m pip install 'eubar[yaml]'`
(or `python -m pip install -e '.[yaml]'` from a source checkout).

## An SNV job

Save this as `snv.yaml` in the tutorial directory, alongside `data/` and
`results/`:

```yaml
task: snv
inputs:
  intensities: results/GABPA_MCF7_intensities.tsv
  array: results/MCF7_DNase_8mer.txt
  genome: data/hg38.fa
  snv_list:
    - "chr5:1295113:C>T"
    - "chr5:1295135:C>T"
parameters:
  kmer_size: 8
  best_pval: true
  holm: true
  diagnostics: true
  jobs: 4
output:
  stdout: results/tert_best.tsv
```

```bash
eubar run snv.yaml --dry-run
eubar run snv.yaml
```

`--dry-run` displays the command without running the analysis. It does not
verify that every data file can be read. `output.stdout` captures the result
table; progress messages still go to stderr.

For a variant file, replace `snv_list` with `variants: data/snvs.txt` (or
`snv_list_file: data/snvs.txt`). Use only one variant-input form.

Paths inside a YAML document resolve relative to **that document's directory**.
CLI paths resolve from your working directory. Explicit CLI overrides take
precedence over YAML, which takes precedence over defaults:

```bash
eubar run snv.yaml --jobs 2
```

For a masked job, remove `inputs.array` and add a quoted string such as
`mask: "111100001111"` under `parameters`. Quoting is important, especially for
masks with leading zeros. To use a chosen probe cap, add `max_probes: 800` and
`seed: 1`; the value should come from your assessment, not this syntax example.

## Data preparation tasks

Save as `array.yaml`:

```yaml
task: array
inputs:
  bed: data/MCF7_DNase.bed
  genome: data/hg38.fa
parameters:
  kmer_size: 8
output:
  file: results/MCF7_DNase_8mer.txt
```

Save as `intensities.yaml`:

```yaml
task: intensities
inputs:
  bed: data/MCF7_DNase.bed
  signal: data/GABPA_MCF7.bw
  genome_fasta: data/hg38.fa
parameters:
  summary: max
output:
  file: results/GABPA_MCF7_intensities.tsv
```

The intensity FASTA key is `genome_fasta`; array, SNV, scan and masked motif
jobs use `genome`. Output keys also depend on the task:

| Task | Main output keys |
| --- | --- |
| `array`, `intensities` | `file` |
| `snv`, `scan` | `stdout` |
| `motifs` | `directory`, `prefix` |
| `calibrate_max_probes`, `calibrate_mask` | `prefix` |

`stdout` can additionally capture console output for other tasks. For scan,
`save_figure` belongs under `parameters`, not `output`.

## Calibration tasks

Save as `calibrate_max_probes.yaml`:

```yaml
task: calibrate_max_probes
inputs:
  intensities: results/GABPA_MCF7_intensities.tsv
  array: results/MCF7_DNase_8mer.txt
parameters:
  kmer_size: 8
  caps: [100, 200, 400, 600, 800, 1000, 1250, 1500, 2000, 2500]
  n_families: 100
  repeats: 50
  seed: 1
  plot: true
output:
  prefix: results/GABPA_maxprobes
```

Save as `calibrate_mask.yaml`:

```yaml
task: calibrate_mask
inputs:
  intensities: results/GABPA_MCF7_intensities.tsv
  genome: data/hg38.fa
parameters:
  masks: ["11111111", "111100001111", "11111000011111"]
  stability_probes: 100
  n_families: 500
  repeats: 50
  random_controls: 20
  seed: 1
  plot: true
output:
  prefix: results/GABPA_masks
```

These examples match the CLI settings in [Masks and calibration](06_masks_and_calibration.md).
For mask files, use `inputs.mask_file` and `inputs.control_mask_file`. Inline
`masks` and `control_masks` belong under `parameters`. Read the reports and then
set the chosen mask or cap in your analysis YAML; recommendations are not applied
automatically.

## Pipelines

Once the three task files above exist, save `pipeline.yaml`:

```yaml
task: pipeline
steps:
  - array.yaml
  - intensities.yaml
  - snv.yaml
```

```bash
eubar run pipeline.yaml
```

Steps run sequentially. A nonzero return status stops the pipeline; recursive
cycles are rejected. Each step's paths are relative to its own YAML file.
This is not a caching workflow: rerunning the pipeline executes its steps again.
Do not include an expensive calibration step on every run unless you intend
to repeat it.

## Parameter spelling

Most keys use the CLI name with hyphens replaced by underscores:
`best_pval`, `max_probes`, `rand_n`, `mask_candidate_probes`. Keep the capital
`F` in motif `min_F`. Motif `seed` is a sequence string; SNV, scan and calibration
`seed` is a numeric sampling seed. Flags use YAML `true` or `false`.

Unknown keys are rejected. The [Python API](05_python_api.md) uses the same task
execution functions, with a few field-name differences described there.
