# EUBAR

EUBAR is a small toolkit for:

- building k-mer → region index files (`array`)
- computing region-level intensities from signal tracks (`intensities`)
- running motif regression in two modes:
  - region scan (`scan`)
  - SNV list (`snv`)
- motif discovery (`motifs`)

The main interface is a single umbrella command:

- `eubar <command> [args...]`

---

## Install

```bash
pip install .

eubar --help
# subcommands:
#   eubar array ...
#   eubar intensities ...
#   eubar scan ...
#   eubar snv ...
#   eubar motifs ...
```
