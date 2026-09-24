from __future__ import annotations

from pathlib import Path

import pytest

from eubar.config import ConfigError, build_argv, load_yaml, pipeline_step_paths
from eubar.run import _parse_runner_args


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_snv_yaml_builds_existing_cli_and_resolves_paths(tmp_path):
    cfg_path = _write(
        tmp_path / "snv.yaml",
        """
task: snv
inputs:
  intensities: data/tf.int
  array: data/array.txt
  genome: ref/hg38.fa
  variants: variants.txt
parameters:
  best_pval: true
  holm: true
  jobs: 4
  mask: "11111000011111"
output:
  stdout: out/results.tsv
""",
    )
    cfg = load_yaml(cfg_path)
    command, argv, stdout = build_argv(cfg)
    assert command == "snv"
    assert argv[argv.index("--jobs") + 1] == "4"
    assert "--best-pval" in argv
    assert "--holm" in argv
    assert argv[argv.index("--mask") + 1] == "11111000011111"
    assert argv[argv.index("--intensities") + 1] == str((tmp_path / "data/tf.int").resolve())
    assert argv[argv.index("--snv-list-file") + 1] == str((tmp_path / "variants.txt").resolve())
    assert stdout == str((tmp_path / "out/results.tsv").resolve())


def test_mask_lists_are_serialized_as_csv(tmp_path):
    cfg_path = _write(
        tmp_path / "mask.yaml",
        """
task: calibrate_mask
inputs:
  intensities: tf.int
  genome: hg38.fa
parameters:
  masks: ["1111111111", "11111000011111"]
  control_masks: ["11011011011011"]
  random_controls: 20
output:
  prefix: mask_test
""",
    )
    cfg = load_yaml(cfg_path)
    command, argv, _ = build_argv(cfg)
    assert command == "calibrate"
    assert argv[0] == "mask"
    assert argv[argv.index("--masks") + 1] == "1111111111,11111000011111"
    assert argv[argv.index("--control-masks") + 1] == "11011011011011"


def test_max_probe_caps_list_is_serialized(tmp_path):
    cfg_path = _write(
        tmp_path / "max.yaml",
        """
task: calibrate_max_probes
inputs:
  intensities: tf.int
  array: array.txt
parameters:
  caps: [100, 200, 400]
output:
  prefix: result
""",
    )
    command, argv, _ = build_argv(load_yaml(cfg_path))
    assert command == "calibrate"
    assert argv[:1] == ["max-probes"]
    assert argv[argv.index("--caps") + 1] == "100,200,400"


def test_pipeline_paths_are_relative_to_pipeline_yaml(tmp_path):
    (tmp_path / "configs").mkdir()
    cfg_path = _write(
        tmp_path / "configs" / "pipeline.yaml",
        """
task: pipeline
steps:
  - array.yaml
  - snv.yaml
""",
    )
    paths = pipeline_step_paths(load_yaml(cfg_path))
    assert paths == [
        (tmp_path / "configs/array.yaml").resolve(),
        (tmp_path / "configs/snv.yaml").resolve(),
    ]


def test_unknown_task_key_fails_fast(tmp_path):
    cfg_path = _write(
        tmp_path / "bad.yaml",
        """
task: snv
inputs:
  intensities: tf.int
  genome: hg38.fa
  variants: variants.txt
parameters:
  jobz: 4
""",
    )
    with pytest.raises(ConfigError, match="jobz"):
        load_yaml(cfg_path)


def test_runner_allows_single_config_cli_override():
    configs, overrides, dry_run, print_argv = _parse_runner_args(
        ["snv.yaml", "--jobs", "8"]
    )
    assert configs == ["snv.yaml"]
    assert overrides == ["--jobs", "8"]
    assert dry_run is False
    assert print_argv is False



def _template_dir() -> Path:
    # Support both a normal source tree (repo/eubar/templates) and the flat
    # development checkout used by the project maintainer (repo/templates).
    root = Path(__file__).resolve().parents[1]
    nested = root / "eubar" / "templates"
    return nested if nested.exists() else root / "templates"


def test_intensities_template_hides_advanced_residualization_controls():
    text = (_template_dir() / "intensities.yaml").read_text(encoding="utf-8")
    assert "resid_use_length:" not in text
    assert "resid_open:" not in text
    assert "resid_output:" not in text
    assert "keep_temp:" not in text


def test_public_templates_hide_legacy_or_developer_switches():
    hidden_keys = {"raw_lp:", "no_covariates:", "no_rand:", "debug:", "mode:", "pooled:"}
    for name in ("snv.yaml", "scan.yaml", "calibrate_max_probes.yaml", "calibrate_mask.yaml"):
        text = (_template_dir() / name).read_text(encoding="utf-8")
        for key in hidden_keys:
            assert key not in text, f"{key} unexpectedly exposed in {name}"



def test_hidden_intensity_controls_remain_configurable(tmp_path):
    cfg_path = _write(
        tmp_path / "intensities_advanced.yaml",
        """
task: intensities
inputs:
  bed: regions.bed
  signal: signal.bigWig
  genome_fasta: hg38.fa
parameters:
  resid_use_length: true
  resid_open: force
  resid_output: intensity_like
  keep_temp: true
output:
  file: tf.int
""",
    )
    command, argv, _ = build_argv(load_yaml(cfg_path))
    assert command == "intensities"
    assert "--resid-use-length" in argv
    assert argv[argv.index("--resid-open") + 1] == "force"
    assert argv[argv.index("--resid-output") + 1] == "intensity_like"
    assert "--keep-temp" in argv


def test_hidden_snv_controls_remain_configurable(tmp_path):
    cfg_path = _write(
        tmp_path / "snv_advanced.yaml",
        """
task: snv
inputs:
  intensities: tf.int
  array: array.txt
  genome: hg38.fa
  snv_list: ["chr1:10:A>G"]
parameters:
  raw_lp: true
  no_covariates: true
  mode: nb
output:
  stdout: out.tsv
""",
    )
    command, argv, _ = build_argv(load_yaml(cfg_path))
    assert command == "snv"
    assert "--raw-lp" in argv
    assert "--no-covariates" in argv
    assert argv[argv.index("--mode") + 1] == "nb"

def test_bundled_templates_all_parse(tmp_path):
    template_dir = _template_dir()
    for source in sorted(template_dir.glob("*.yaml")):
        dest = tmp_path / source.name
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        cfg = load_yaml(dest)
        assert cfg.task
        if cfg.task != "pipeline":
            command, argv, _ = build_argv(cfg)
            assert command
            assert isinstance(argv, list)


def test_calibration_templates_hide_implementation_thresholds():
    hidden = {
        "candidate_multiplier:",
        "max_effect_sd:",
        "max_effect_bias:",
        "min_eligible_fraction:",
        "borderline_relative_tolerance:",
        "near_optimal_relative_tolerance:",
    }
    for name in ("calibrate_max_probes.yaml", "calibrate_mask.yaml"):
        text = (_template_dir() / name).read_text(encoding="utf-8")
        for key in hidden:
            assert key not in text, f"{key} unexpectedly exposed in {name}"
