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

From the repo root:

```bash
pip install -e .
```
