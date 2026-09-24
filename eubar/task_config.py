"""Task inputs and defaults shared by Python, CLI and YAML callers."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Optional, Tuple, get_type_hints, get_args


class TaskConfigError(ValueError):
    """Invalid task inputs, reported as usage errors by the CLI."""


class TaskOptions:
    _required: ClassVar[tuple] = ()
    _choices: ClassVar[dict] = {}

    @classmethod
    def from_values(cls, values):
        """Convert adapter values without parsing a command line."""
        hints = get_type_hints(cls)
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise TaskConfigError('unknown option(s): ' + ', '.join(sorted(unknown)))
        converted = {}
        for name, value in values.items():
            if value is None:
                converted[name] = None
                continue
            kind = hints[name]
            if type(None) in get_args(kind):
                kind = next(t for t in get_args(kind) if t is not type(None))
            try:
                if name == 'caps':
                    from .core.calibration import parse_caps
                    value = parse_caps(value if isinstance(value, str) else ','.join(map(str, value)))
                    value = tuple(value)
                elif kind is bool:
                    if not isinstance(value, bool):
                        raise TaskConfigError('expected true or false')
                elif kind is int:
                    value = int(str(value))
                elif kind is float:
                    value = float(str(value))
                elif kind is str:
                    value = str(value)
            except (TypeError, ValueError) as exc:
                raise TaskConfigError(f'{name}: {exc}') from exc
            converted[name] = value
        return cls(**converted)

    def validate(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            value = getattr(self, field.name)
            kind = hints[field.name]
            if type(None) in get_args(kind):
                if value is None:
                    continue
                kind = next(t for t in get_args(kind) if t is not type(None))
            if kind in (str, int, float, bool):
                valid = isinstance(value, kind)
                if kind is float:
                    valid = isinstance(value, (int, float))
                if kind in (int, float) and isinstance(value, bool):
                    valid = False
                if not valid:
                    raise TaskConfigError(f'{field.name} must be {kind.__name__}')
        for name in self._required:
            if getattr(self, name) is None:
                raise TaskConfigError(f'{name} is required')
        for name, choices in self._choices.items():
            value = getattr(self, name)
            if value not in choices:
                raise TaskConfigError(f'{name} must be one of {choices}')


@dataclass(frozen=True)
class ArrayConfig(TaskOptions):
    bed: Optional[str] = None
    genome: Optional[str] = None
    kmer_size: int = 8
    output: Optional[str] = None

    _required = ('bed', 'genome', 'output')


@dataclass(frozen=True)
class IntensitiesConfig(TaskOptions):
    bed: Optional[str] = None
    signal: Optional[str] = None
    output: Optional[str] = None
    genome_size_file: Optional[str] = None
    keep_temp: bool = False
    no_residualize: bool = False
    genome: Optional[str] = None
    resid_use_length: bool = False
    resid_open: str = 'off'
    resid_output: str = 'resid_log'
    summary: str = 'max'
    window_bp: int = 100

    _required = ('bed', 'signal', 'output')
    _choices = {'resid_open': ('auto', 'off', 'force'), 'resid_output': ('resid_log', 'log_corrected', 'intensity_like'), 'summary': ('max', 'mean', 'center_max', 'center_mean')}

    def validate(self):
        super().validate()
        if not self.no_residualize and not self.genome:
            raise TaskConfigError('--genome-fasta is required (use --no-residualize to skip residualization).')


@dataclass(frozen=True)
class SnvConfig(TaskOptions):
    intensities: Optional[str] = None
    array: Optional[str] = None
    genome: Optional[str] = None
    snv_list: Optional[str] = None
    snv_list_file: Optional[str] = None
    kmer_size: int = 8
    mask: Optional[str] = None
    rand_n: int = 500
    no_rand: bool = False
    best_pval: bool = False
    holm: bool = False
    diagnostics: bool = False
    mode: str = 'ols'
    no_covariates: bool = False
    raw_lp: bool = False
    seed: int = 0
    max_probes: Optional[int] = None
    jobs: int = 1
    debug: bool = False

    _required = ('intensities', 'genome')
    _choices = {'mode': ('nb', 'ols')}

    def validate(self):
        super().validate()
        if self.diagnostics and not self.best_pval:
            raise TaskConfigError('--diagnostics requires --best_pval')
        if self.jobs < 1:
            raise TaskConfigError('--jobs must be >= 1')
        if self.kmer_size < 2:
            raise TaskConfigError('--kmer-size must be >= 2')
        if self.rand_n < 1 and not self.no_rand:
            raise TaskConfigError('--rand-n must be >= 1 unless --no-rand is used')
        if self.mask is None and not self.array:
            raise TaskConfigError('--array is required unless --mask is supplied')
        if bool(self.snv_list) == bool(self.snv_list_file):
            raise TaskConfigError('supply one of snv_list or snv_list_file')
        if self.mask is not None:
            from .core.patterns import SequenceMask
            try:
                SequenceMask.parse(self.mask)
            except ValueError as exc:
                raise TaskConfigError(str(exc)) from exc


@dataclass(frozen=True)
class ScanConfig(TaskOptions):
    intensities: Optional[str] = None
    array: Optional[str] = None
    genome: Optional[str] = None
    region: Optional[str] = None
    kmer_size: int = 8
    mode: str = 'ols'
    pooled: bool = False
    best_pval: bool = False
    no_covariates: bool = False
    reverse: bool = False
    raw_lp: bool = False
    seed: int = 0
    max_probes: Optional[int] = None
    save_figure: Optional[str] = None
    holm: bool = False
    rand_n: int = 500
    no_rand: bool = False
    diagnostics: bool = False

    _required = ('intensities', 'array', 'genome', 'region')
    _choices = {'mode': ('ols', 'nb')}

    def validate(self):
        super().validate()
        if self.pooled and self.best_pval:
            raise TaskConfigError('--pooled cannot be used with --best-pval')
        if self.holm and not self.best_pval:
            raise TaskConfigError('--holm requires --best-pval and cannot be used with --pooled')
        if not self.holm and (self.diagnostics or self.no_rand or self.rand_n != 500):
            raise TaskConfigError('--diagnostics, --no-rand and custom --rand-n require --best-pval --holm')
        if self.kmer_size < 1 or self.rand_n < 0:
            raise TaskConfigError('--kmer-size must be positive and --rand-n must be non-negative')


@dataclass(frozen=True)
class MotifsConfig(TaskOptions):
    intensities: Optional[str] = None
    array: Optional[str] = None
    mask: Optional[str] = None
    genome: Optional[str] = None
    mask_candidate_probes: int = 5000
    mask_max_candidates: int = 10000
    kmer_size: int = 8
    combine_revcomp: bool = True
    min_F: int = 20
    max_gaps: int = 3
    min_per_base: int = 20
    min_support: int = 1
    pseudocount: float = 0.0
    beta: float = 10.0
    auto_max_steps: int = 20
    extend_left: int = -1
    extend_right: int = -1
    ic_stop_threshold: float = 0.2
    ic_stop_consecutive: int = 2
    outdir: str = '.'
    prefix: str = 'affinity_motif'
    pretty_logo: bool = False
    seed: Optional[str] = None
    top_n_report: int = 50

    _required = ('intensities',)


@dataclass(frozen=True)
class CalibrationConfig(TaskOptions):
    calibration: str = 'max-probes'
    array: Optional[str] = None
    intensities: Optional[str] = None
    out_prefix: Optional[str] = None
    genome: Optional[str] = None
    masks: Optional[str] = None
    mask_file: Optional[str] = None
    control_masks: Optional[str] = None
    control_mask_file: Optional[str] = None
    random_controls: int = 0
    stability_probes: int = 100
    candidate_multiplier: int = 5
    kmer_size: int = 8
    n_families: Optional[int] = None
    repeats: int = 50
    caps: Tuple[int, ...] = (100, 200, 400, 600, 800, 1000, 1250, 1500, 2000, 2500)
    seed: int = 1
    min_probes: int = 50
    no_covariates: bool = False
    raw_lp: bool = False
    max_effect_sd: float = 0.1
    max_effect_bias: float = 0.1
    min_eligible_fraction: float = 0.5
    borderline_relative_tolerance: float = 0.05
    near_optimal_relative_tolerance: float = 0.05
    plot: bool = False
    quiet: bool = False

    _required = ('intensities', 'out_prefix')
    _choices = {'calibration': ('max-probes', 'mask')}
