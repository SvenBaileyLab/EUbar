# Masks, probe limits and calibration

Masks choose which sequence positions must match. Probe limits choose how many
matching probes enter a fit. Calibration helps assess representation and
stability before selecting either setting.

## Reading a mask

A mask is a string of `1`s and `0`s. A `1` marks an informative position whose
base is matched; a `0` marks a spacer whose base is ignored.

| Mask | Span | Informative bases | Meaning |
| --- | --- | --- | --- |
| `11111111` | 8 bp | 8 | Eight adjacent matched positions |
| `111100001111` | 12 bp | 8 | Two four-base blocks separated by four ignored bases |
| `11111000011111` | 14 bp | 10 | Two five-base blocks separated by four ignored bases |

Spacing is fixed. Ignoring four bases does not allow an arbitrary-length gap or
an insertion/deletion. Explicit masks must contain at least two and at most 31
informative positions. Asymmetric masks are accepted; matching includes reverse
complements.

For masked SNVs, the variant is tested only where it falls on a `1`. An eight-one
mask therefore provides eight candidate alignments, even if its span is twelve
bases. In SNV output, `x` marks an ignored position and `.` marks the tested base.
`motif_pos` remains a window-start offset, so masked positions need not be
consecutive.

Masked SNV and motif tasks obtain probe coordinates from the intensity file and
sequence from `--genome`. They do not use `--array`, even if one is supplied.
The mask span replaces `--kmer-size`. An explicit all-ones mask still uses the
masked matcher; do not assume it reproduces the ordinary array-based path
exactly. Scan currently has no mask option.

More informative positions impose a more specific match and can reduce probe
support. Fewer informative positions pool more sequence contexts. A longer span
can represent separated blocks, but a mask alone does not demonstrate that a TF
has a bipartite motif. Compare support, stability and biological interpretation.

## What `--max-probes` limits

SNV and non-pooled scan analyses use all matching probes by default. A positive
`--max-probes N` requests an approximate budget of matched probes for each
window's fit, allocated proportionally across allele groups. Sampling is
without replacement within each group.

The implementation rounds each group's allocation and retains at least one
probe per nonempty group. Consequently, the realized total can slightly exceed
`N`; it is not a strict global row limit. It is also not `N` probes per allele.
Later filtering can reduce the usable fit size.

AFF and RAND use separate deterministic subsamples. In RAND, the cap applies
to the matched allele probes; background probes are added separately and are
controlled by `--rand-n` (default 500). This is why RAND `n_probes` can exceed
the cap by much more than rounding alone.

`--seed` controls matched-probe subsampling and defaults to 0 in SNV/scan.
Changing it does not resample the RAND background, which retains its own
deterministic scheme. Keep the input files, options and seed fixed when
comparing runs. Use `--diagnostics` with `--best-pval` to inspect realized counts.

A lower cap can reduce fitting cost but changes the data entering the model,
including effect estimates and p-values. It does not correct inflated p-values
or establish that the regression assumptions hold. Index construction and
sequence matching may still dominate runtime.

## Calibrate a contiguous probe cap

```bash
eubar calibrate max-probes \
  --intensities results/GABPA_MCF7_intensities.tsv \
  --array results/MCF7_DNase_8mer.txt \
  --kmer-size 8 \
  --caps 100,200,400,600,800,1000,1250,1500,2000,2500 \
  --n-families 100 --repeats 50 --seed 1 \
  --out-prefix results/GABPA_maxprobes --plot
```

A *family* contains sequence contexts differing at one tested base. Calibration
compares allele contrasts fitted with all usable family probes against repeated
subsamples at each cap. It uses OLS and evaluates effect stability, not a target
p-value. This command uses contiguous array families; it does not calibrate a
specific explicit mask.

| Output suffix | Contents |
| --- | --- |
| `.summary.tsv` | Representation and stability at each tested cap |
| `.family_curves.tsv` | Stability by family and cap, including successful repeats |
| `.contrasts.tsv` | Full-data allele contrasts |
| `.pvalue_quartiles.tsv` | Diagnostic p-value summaries grouped by effect size |
| `.recommendation.tsv` | Selected cap, status and selection criteria |
| `.pdf` | Diagnostic plots, when `--plot` is supplied |

Read `.recommendation.tsv` alongside `.summary.tsv`:

- `eligible_fraction` is the fraction of accepted families with enough probes
  for that cap.
- `effect_sd_norm` measures variation across repeated subsample effects, divided
  by the family's intensity standard deviation.
- `effect_bias_norm` measures the absolute difference between the mean subsample
  effect and the full-data effect, on the same normalized scale.
- Family scores summarize available allele contrasts by their median. The
  summary's `_p90` columns are the 90th percentiles across evaluated families.

The default rule selects the smallest tested cap with eligible fraction at
least 0.5, p90 normalized effect SD at most 0.1, and p90 normalized effect bias
at most 0.1. These are stability criteria, not accuracy estimates against
experimental truth. P-values do not enter the recommendation.

If status is `NO_TESTED_CAP_MET_STABILITY_CRITERIA`, there is no recommended
cap. `best_tested_cap` is a diagnostic fallback, not a passing result. A
borderline annotation on the preceding cap does not change the selection rule.

Apply a chosen cap explicitly, for example by adding `--max-probes 800 --seed 1`
to an SNV command **if 800 is the value you decided to use**. Calibration never
changes subsequent jobs automatically.

## Compare candidate masks

```bash
eubar calibrate mask \
  --intensities results/GABPA_MCF7_intensities.tsv \
  --genome data/hg38.fa \
  --masks 11111111,111100001111,11111000011111 \
  --stability-probes 100 \
  --n-families 500 --repeats 50 --seed 1 \
  --random-controls 20 \
  --out-prefix results/GABPA_masks --plot
```

These masks illustrate candidate comparison; they are not established GABPA
settings. Candidates can instead be supplied in `--mask-file`, with one mask
per line. Quote masks in YAML to preserve leading zeros.

Each mask is evaluated at the same `--stability-probes` count. A family must
have at least `max(min_probes, stability_probes + 1)` usable probes, at least
two alleles, and a usable full-data fit with finite contrasts. The reported
eligible fraction uses the sampled candidate-family pool as its denominator;
it is not coverage of every possible genomic context.

Among candidate masks meeting the default minimum eligible fraction of 0.5
and having evaluated families with finite stability scores, the recommendation
chooses the lowest p90 normalized effect SD. Ties are resolved by residual noise,
then representation, then mask string. Masks within 5% of the best score are
reported as near-optimal alternatives. Inspect those alternatives and support
counts rather than treating a tiny score difference as a clear biological win.

| Output suffix | Contents |
| --- | --- |
| `.summary.tsv` | One row per candidate or control mask, with support, stability and ranks |
| `.families.tsv` | Family counts, allele representation, eligibility and residual noise |
| `.family_curves.tsv` | Repeated-subsample stability for evaluated families |
| `.recommendation.tsv` | Candidate recommendation, alternatives and control comparison |
| `.pdf` | Diagnostic plots, when requested |

`NO_CANDIDATE_MASK_MET_REPRESENTATION_CRITERIA` means no candidate passed.
`RECOMMENDED_WITH_NEAR_OPTIMAL_ALTERNATIVES` means several candidates are within
the tolerance band. `residual_sd_ratio` describes fit residual SD relative to
family intensity SD; smaller values indicate less residual variation.

### Control masks

`--control-masks` and `--control-mask-file` add explicit diagnostic controls.
They are evaluated but cannot become the automatic recommendation.

`--random-controls N` generates controls matched to the recommended candidate's
span and number of informative positions, keeping the first and last positions
informative. The feasible distinct patterns can limit how many controls are
available. Check the requested and evaluated counts in the report.

The control comparison asks whether the candidate's stability is unusual among
those patterns. A control can outperform the recommended candidate without
replacing it. `random_control_empirical_p` is a descriptive rank comparison,
`(1 + controls as good or better) / (1 + evaluated controls)`, not a validation
p-value for TF binding or proof of motif specificity.

Mask calibration does not select a production probe cap. `--stability-probes`
sets the comparison size for this diagnostic only. Use the chosen mask explicitly
in later SNV or motif commands, and assess the support for those analyses.

## Calibration settings

| Option | Default | Use |
| --- | --- | --- |
| `--n-families` | 100 for max-probes; 500 for masks | Requested number of evaluated families; available data may yield fewer |
| `--repeats` | 50 | Subsamples per family and tested size |
| `--seed` | 1 | Reproducible calibration sampling |
| `--min-probes` | 50 | Minimum usable family size; masks also require more than `stability_probes` |
| `--caps` | 100 through 2500 at the values above | Sizes tested by max-probes calibration |
| `--stability-probes` | 100 | Fixed comparison size for masks |
| `--random-controls` | 0 | Requested random mask controls |
| `--plot` | Off | Write a PDF alongside tables |
| `--quiet` | Off | Reduce progress messages |

Advanced options retained for reproducibility include `--max-effect-sd` (0.1),
`--max-effect-bias` (0.1), `--min-eligible-fraction` (0.5),
`--borderline-relative-tolerance` (0.05; caps),
`--near-optimal-relative-tolerance` (0.05; masks), and
`--candidate-multiplier` (5; masks). Changing thresholds changes the decision
rule; record them with the results. `--no-covariates` and `--raw-lp` also alter
the fitted design.

**Related:** [SNV analysis](02_snv.md) · [Motif discovery](04_motifs.md) · [YAML jobs](07_yaml.md)
