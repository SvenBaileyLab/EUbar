from __future__ import annotations

"""YAML configuration support for EUbar.

YAML documents resolve to the same task options used by the CLI and Python API.
Argument rendering is retained for dry-run display and compatibility.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


class ConfigError(ValueError):
    """Raised for an invalid EUbar YAML configuration."""


@dataclass(frozen=True)
class OptionSpec:
    cli: str
    kind: str = "value"  # value | flag | csv | path


@dataclass(frozen=True)
class TaskSpec:
    command: str
    positional: Tuple[str, ...]
    inputs: Mapping[str, OptionSpec]
    parameters: Mapping[str, OptionSpec]
    output: Mapping[str, OptionSpec]


@dataclass
class TaskConfig:
    source: Path
    task: str
    raw: Dict[str, Any]
    spec: Optional[TaskSpec] = None

    @property
    def base_dir(self) -> Path:
        return self.source.parent


PATH = "path"
VALUE = "value"
FLAG = "flag"
CSV = "csv"


def _opt(cli: str, kind: str = VALUE) -> OptionSpec:
    return OptionSpec(cli=cli, kind=kind)


TASK_SPECS: Dict[str, TaskSpec] = {
    "array": TaskSpec(
        command="array",
        positional=(),
        inputs={
            "bed": _opt("--bed", PATH),
            "genome": _opt("--genome", PATH),
        },
        parameters={"kmer_size": _opt("--kmer-size")},
        output={
            "file": _opt("--output", PATH),
            "stdout": _opt("", PATH),
        },
    ),
    "intensities": TaskSpec(
        command="intensities",
        positional=(),
        inputs={
            "bed": _opt("--bed", PATH),
            "signal": _opt("--signal", PATH),
            "genome_fasta": _opt("--genome-fasta", PATH),
            "genome_size_file": _opt("--genome-size-file", PATH),
        },
        parameters={
            "keep_temp": _opt("--keep-temp", FLAG),
            "no_residualize": _opt("--no-residualize", FLAG),
            "resid_use_length": _opt("--resid-use-length", FLAG),
            "resid_open": _opt("--resid-open"),
            "resid_output": _opt("--resid-output"),
            "summary": _opt("--summary"),
            "window_bp": _opt("--window-bp"),
        },
        output={
            "file": _opt("--output", PATH),
            "stdout": _opt("", PATH),
        },
    ),
    "scan": TaskSpec(
        command="scan",
        positional=(),
        inputs={
            "intensities": _opt("--intensities", PATH),
            "array": _opt("--array", PATH),
            "genome": _opt("--genome", PATH),
            "region": _opt("--region"),
        },
        parameters={
            "kmer_size": _opt("--kmer-size"),
            "mode": _opt("--mode"),
            "pooled": _opt("--pooled", FLAG),
            "best_pval": _opt("--best-pval", FLAG),
            "no_covariates": _opt("--no-covariates", FLAG),
            "reverse": _opt("--reverse", FLAG),
            "raw_lp": _opt("--raw-lp", FLAG),
            "seed": _opt("--seed"),
            "max_probes": _opt("--max-probes"),
            "save_figure": _opt("--save-figure", PATH),
            "holm": _opt("--holm", FLAG),
            "rand_n": _opt("--rand-n"),
            "no_rand": _opt("--no-rand", FLAG),
            "diagnostics": _opt("--diagnostics", FLAG),
        },
        output={"stdout": _opt("", PATH)},
    ),
    "snv": TaskSpec(
        command="snv",
        positional=(),
        inputs={
            "intensities": _opt("--intensities", PATH),
            "array": _opt("--array", PATH),
            "genome": _opt("--genome", PATH),
            "variants": _opt("--snv-list-file", PATH),
            "snv_list_file": _opt("--snv-list-file", PATH),
            "snv_list": _opt("--snv-list", CSV),
        },
        parameters={
            "kmer_size": _opt("--kmer-size"),
            "mask": _opt("--mask"),
            "rand_n": _opt("--rand-n"),
            "no_rand": _opt("--no-rand", FLAG),
            "best_pval": _opt("--best-pval", FLAG),
            "holm": _opt("--holm", FLAG),
            "diagnostics": _opt("--diagnostics", FLAG),
            "mode": _opt("--mode"),
            "no_covariates": _opt("--no-covariates", FLAG),
            "raw_lp": _opt("--raw-lp", FLAG),
            "seed": _opt("--seed"),
            "max_probes": _opt("--max-probes"),
            "jobs": _opt("--jobs"),
            "debug": _opt("--debug", FLAG),
        },
        output={"stdout": _opt("", PATH)},
    ),
    "motifs": TaskSpec(
        command="motifs",
        positional=(),
        inputs={
            "intensities": _opt("--intensities", PATH),
            "array": _opt("--array", PATH),
            "genome": _opt("--genome", PATH),
        },
        parameters={
            "mask": _opt("--mask"),
            "mask_candidate_probes": _opt("--mask-candidate-probes"),
            "mask_max_candidates": _opt("--mask-max-candidates"),
            "kmer_size": _opt("--kmer-size"),
            "no_combine_revcomp": _opt("--no-combine-revcomp", FLAG),
            "min_F": _opt("--min-F"),
            "max_gaps": _opt("--max-gaps"),
            "min_per_base": _opt("--min-per-base"),
            "min_support": _opt("--min-support"),
            "pseudocount": _opt("--pseudocount"),
            "beta": _opt("--beta"),
            "auto_max_steps": _opt("--auto-max-steps"),
            "extend_left": _opt("--extend-left"),
            "extend_right": _opt("--extend-right"),
            "ic_stop_threshold": _opt("--ic-stop-threshold"),
            "ic_stop_consecutive": _opt("--ic-stop-consecutive"),
            "pretty_logo": _opt("--pretty-logo", FLAG),
            "seed": _opt("--seed"),
            "top_n_report": _opt("--top-n-report"),
        },
        output={
            "directory": _opt("--outdir", PATH),
            "prefix": _opt("--prefix"),
            "stdout": _opt("", PATH),
        },
    ),
    "calibrate_max_probes": TaskSpec(
        command="calibrate",
        positional=("max-probes",),
        inputs={
            "intensities": _opt("--intensities", PATH),
            "array": _opt("--array", PATH),
        },
        parameters={
            "kmer_size": _opt("--kmer-size"),
            "n_families": _opt("--n-families"),
            "repeats": _opt("--repeats"),
            "caps": _opt("--caps", CSV),
            "seed": _opt("--seed"),
            "min_probes": _opt("--min-probes"),
            "no_covariates": _opt("--no-covariates", FLAG),
            "raw_lp": _opt("--raw-lp", FLAG),
            "max_effect_sd": _opt("--max-effect-sd"),
            "max_effect_bias": _opt("--max-effect-bias"),
            "min_eligible_fraction": _opt("--min-eligible-fraction"),
            "borderline_relative_tolerance": _opt("--borderline-relative-tolerance"),
            "plot": _opt("--plot", FLAG),
            "quiet": _opt("--quiet", FLAG),
        },
        output={
            "prefix": _opt("--out-prefix", PATH),
            "stdout": _opt("", PATH),
        },
    ),
    "calibrate_mask": TaskSpec(
        command="calibrate",
        positional=("mask",),
        inputs={
            "intensities": _opt("--intensities", PATH),
            "genome": _opt("--genome", PATH),
            "mask_file": _opt("--mask-file", PATH),
            "control_mask_file": _opt("--control-mask-file", PATH),
        },
        parameters={
            "masks": _opt("--masks", CSV),
            "control_masks": _opt("--control-masks", CSV),
            "random_controls": _opt("--random-controls"),
            "stability_probes": _opt("--stability-probes"),
            "candidate_multiplier": _opt("--candidate-multiplier"),
            "kmer_size": _opt("--kmer-size"),
            "n_families": _opt("--n-families"),
            "repeats": _opt("--repeats"),
            "seed": _opt("--seed"),
            "min_probes": _opt("--min-probes"),
            "no_covariates": _opt("--no-covariates", FLAG),
            "raw_lp": _opt("--raw-lp", FLAG),
            "min_eligible_fraction": _opt("--min-eligible-fraction"),
            "near_optimal_relative_tolerance": _opt("--near-optimal-relative-tolerance"),
            "plot": _opt("--plot", FLAG),
            "quiet": _opt("--quiet", FLAG),
        },
        output={
            "prefix": _opt("--out-prefix", PATH),
            "stdout": _opt("", PATH),
        },
    ),
}


TASK_ALIASES = {
    "intensity": "intensities",
    "calibrate-max-probes": "calibrate_max_probes",
    "calibrate_max-probes": "calibrate_max_probes",
    "max-probes": "calibrate_max_probes",
    "max_probes": "calibrate_max_probes",
    "calibrate-mask": "calibrate_mask",
    "mask-calibration": "calibrate_mask",
}


def canonical_task(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("YAML must contain a non-empty 'task:' field")
    task = value.strip().lower().replace(" ", "_")
    task = TASK_ALIASES.get(task, task)
    if task == "pipeline":
        return task
    if task not in TASK_SPECS:
        valid = ", ".join(sorted(list(TASK_SPECS) + ["pipeline"]))
        raise ConfigError(f"unknown task {value!r}; expected one of: {valid}")
    return task


def _require_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a YAML mapping/object")
    return dict(value)


def load_yaml(path: os.PathLike) -> TaskConfig:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ConfigError(
            "YAML support requires PyYAML. Install it with: pip install pyyaml"
        ) from exc

    try:
        with source.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except Exception as exc:
        raise ConfigError(f"could not parse YAML {source}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config is empty: {source}")
    if not isinstance(raw, dict):
        raise ConfigError("top level of an EUbar YAML must be a mapping/object")

    allowed_top = {"task", "name", "description", "inputs", "parameters", "output", "steps"}
    unknown_top = sorted(set(raw) - allowed_top)
    if unknown_top:
        raise ConfigError("unknown top-level key(s): " + ", ".join(unknown_top))

    task = canonical_task(raw.get("task"))
    cfg = TaskConfig(source=source, task=task, raw=dict(raw), spec=TASK_SPECS.get(task))
    validate_config(cfg)
    return cfg


def validate_config(cfg: TaskConfig) -> None:
    if cfg.task == "pipeline":
        steps = cfg.raw.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ConfigError("pipeline YAML requires a non-empty 'steps:' list")
        for i, step in enumerate(steps, 1):
            if not isinstance(step, str) or not step.strip():
                raise ConfigError(f"pipeline step {i} must be a YAML filename string")
        for section in ("inputs", "parameters", "output"):
            if cfg.raw.get(section):
                raise ConfigError(f"pipeline YAML does not use '{section}:'")
        return

    assert cfg.spec is not None
    for section_name, allowed in (
        ("inputs", cfg.spec.inputs),
        ("parameters", cfg.spec.parameters),
        ("output", cfg.spec.output),
    ):
        section = _require_mapping(cfg.raw.get(section_name), section_name)
        unknown = sorted(set(section) - set(allowed))
        if unknown:
            raise ConfigError(
                f"unknown {section_name} key(s) for task '{cfg.task}': "
                + ", ".join(unknown)
            )

    if cfg.task == "snv":
        inputs = _require_mapping(cfg.raw.get("inputs"), "inputs")
        supplied = [
            k for k in ("variants", "snv_list_file", "snv_list")
            if inputs.get(k) not in (None, "", [])
        ]
        if len(supplied) > 1:
            raise ConfigError(
                "snv YAML must use only one of inputs.variants, inputs.snv_list_file, or inputs.snv_list"
            )


def _resolve_path(value: Any, base_dir: Path) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"expected a path string, got {type(value).__name__}")
    text = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(text)
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve(strict=False))


def _csv(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def _emit_option(argv: List[str], key: str, value: Any, spec: OptionSpec, base_dir: Path) -> None:
    if value is None:
        return
    if spec.kind == FLAG:
        if not isinstance(value, bool):
            raise ConfigError(f"'{key}' must be true or false")
        if value:
            argv.append(spec.cli)
        return

    if value == "":
        return
    if spec.kind == PATH:
        encoded = _resolve_path(value, base_dir)
    elif spec.kind == CSV:
        encoded = _csv(value)
    else:
        encoded = str(value)
    argv.extend([spec.cli, encoded])


def build_argv(cfg: TaskConfig) -> Tuple[str, List[str], Optional[str]]:
    """Return (command, argv, stdout_path) for a non-pipeline config."""
    if cfg.task == "pipeline":
        raise ConfigError("pipeline configs do not map to a single command argv")
    assert cfg.spec is not None

    argv: List[str] = list(cfg.spec.positional)
    base = cfg.base_dir
    inputs = _require_mapping(cfg.raw.get("inputs"), "inputs")
    params = _require_mapping(cfg.raw.get("parameters"), "parameters")
    output = _require_mapping(cfg.raw.get("output"), "output")

    for key, value in inputs.items():
        _emit_option(argv, key, value, cfg.spec.inputs[key], base)
    for key, value in params.items():
        _emit_option(argv, key, value, cfg.spec.parameters[key], base)

    stdout_path: Optional[str] = None
    for key, value in output.items():
        if key == "stdout":
            if value not in (None, ""):
                stdout_path = _resolve_path(value, base)
            continue
        _emit_option(argv, key, value, cfg.spec.output[key], base)

    return cfg.spec.command, argv, stdout_path


def pipeline_step_paths(cfg: TaskConfig) -> List[Path]:
    if cfg.task != "pipeline":
        raise ConfigError("not a pipeline config")
    steps = cfg.raw.get("steps") or []
    return [Path(_resolve_path(step, cfg.base_dir)) for step in steps]

def task_options(cfg: TaskConfig, overrides: Optional[Mapping[str, Any]] = None):
    """Resolve a YAML document directly to task options and stdout destination."""
    from .api import TASKS
    from .task_config import TaskConfigError

    if cfg.task == 'pipeline':
        raise ConfigError('pipeline configs do not map to a single task')
    values = {}
    stdout_path = None
    for section_name in ('inputs', 'parameters', 'output'):
        specs = getattr(cfg.spec, section_name)
        for key, value in _require_mapping(cfg.raw.get(section_name), section_name).items():
            if value is None or value == '':
                continue
            if section_name == 'output' and key == 'stdout':
                stdout_path = _resolve_path(value, cfg.base_dir)
                continue
            spec = specs[key]
            name = spec.cli.lstrip('-').replace('-', '_')
            if spec.kind == PATH:
                value = _resolve_path(value, cfg.base_dir)
            elif spec.kind == CSV:
                value = _csv(value)
            elif spec.kind == FLAG:
                if not isinstance(value, bool):
                    raise ConfigError(f"'{key}' must be true or false")
            if name == 'genome_fasta':
                name = 'genome'
            elif name == 'no_combine_revcomp':
                name, value = 'combine_revcomp', not value
            values[name] = value
    if cfg.task.startswith('calibrate_'):
        values['calibration'] = cfg.spec.positional[0]
    values.update(overrides or {})
    config_type = TASKS[cfg.task][2]
    try:
        options = config_type.from_values(values)
        options.validate()
    except TaskConfigError as exc:
        raise ConfigError(str(exc)) from exc
    return options, stdout_path
