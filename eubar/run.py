from __future__ import annotations

"""Run one or more EUbar YAML configuration files."""

from contextlib import redirect_stdout
import importlib
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Set, Tuple

from eubar.config import ConfigError, build_argv, load_yaml, pipeline_step_paths, task_options
from eubar.task_config import TaskConfigError


def _quote_argv(command: str, argv: Sequence[str]) -> str:
    try:
        import shlex
        return " ".join(["eubar", command] + [shlex.quote(str(x)) for x in argv])
    except Exception:
        return " ".join(["eubar", command] + [str(x) for x in argv])


def _dispatch(task, options, stdout_path: Optional[str]) -> int:
    from eubar.api import task_runner
    run_task = task_runner(task)

    if stdout_path is None:
        return int(run_task(options) or 0)

    out = Path(stdout_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh, redirect_stdout(fh):
        rc = run_task(options)
    print(f"[eubar run] stdout written to: {out}", file=sys.stderr)
    return int(rc or 0)


def _run_config(
    path: os.PathLike,
    *,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    print_argv: bool = False,
    stack: Optional[Set[Path]] = None,
) -> int:
    cfg = load_yaml(path)
    source = cfg.source
    stack = set() if stack is None else stack
    if source in stack:
        raise ConfigError(f"pipeline cycle detected at: {source}")

    if cfg.task == "pipeline":
        if overrides:
            raise ConfigError(
                "CLI overrides cannot be applied to a pipeline YAML; override an individual task YAML instead"
            )
        child_stack = set(stack)
        child_stack.add(source)
        for step in pipeline_step_paths(cfg):
            print(f"[eubar run] step: {step}", file=sys.stderr)
            rc = _run_config(
                step,
                overrides=(),
                dry_run=dry_run,
                print_argv=print_argv,
                stack=child_stack,
            )
            if rc != 0:
                return rc
        return 0

    if print_argv or dry_run:
        command, argv, stdout_path = build_argv(cfg)
        argv = list(argv) + list(overrides)
        print(_quote_argv(command, argv))
        if stdout_path:
            print(f"  > {stdout_path}")
    if dry_run:
        return 0

    override_values = {}
    if overrides:
        from eubar.cli_options import parse_overrides
        mod = importlib.import_module(f"eubar.{cfg.spec.command}")
        override_values = parse_overrides(mod.build_parser(), overrides)
    options, stdout_path = task_options(cfg, override_values)

    print(f"[eubar run] config: {source}", file=sys.stderr)
    print(f"[eubar run] task: {cfg.task}", file=sys.stderr)
    try:
        return _dispatch(cfg.task, options, stdout_path)
    except ValueError as exc:
        if cfg.task.startswith("calibrate_"):
            raise ConfigError(str(exc)) from exc
        raise


def _parse_runner_args(argv: Sequence[str]) -> Tuple[List[str], List[str], bool, bool]:
    """Parse runner options while allowing raw CLI overrides after one YAML."""
    configs: List[str] = []
    overrides: List[str] = []
    dry_run = False
    print_argv = False

    i = 0
    while i < len(argv):
        token = str(argv[i])
        if token == "--dry-run":
            dry_run = True
            i += 1
            continue
        if token == "--print-argv":
            print_argv = True
            i += 1
            continue
        if token == "--":
            overrides = [str(x) for x in argv[i + 1 :]]
            break
        if token.startswith("-"):
            overrides = [str(x) for x in argv[i:]]
            break
        configs.append(token)
        i += 1

    if not configs:
        raise ConfigError(
            "usage: eubar run CONFIG.yaml [CONFIG2.yaml ...] [--dry-run] [--print-argv] [CLI overrides]"
        )
    if overrides and len(configs) != 1:
        raise ConfigError("CLI overrides are supported only when running a single YAML config")
    return configs, overrides, dry_run, print_argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  eubar run CONFIG.yaml\n"
            "  eubar run A.yaml B.yaml C.yaml\n"
            "  eubar run CONFIG.yaml --jobs 8\n"
            "  eubar run CONFIG.yaml --dry-run\n\n"
            "Paths inside YAML files are resolved relative to that YAML file.\n"
            "When one config is supplied, trailing CLI flags override YAML values."
        )
        return 0

    try:
        configs, overrides, dry_run, print_argv = _parse_runner_args(list(argv))
        for path in configs:
            rc = _run_config(
                path,
                overrides=overrides,
                dry_run=dry_run,
                print_argv=print_argv,
            )
            if rc != 0:
                return rc
        return 0
    except (ConfigError, TaskConfigError) as exc:
        print(f"[eubar run] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
