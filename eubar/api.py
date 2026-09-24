"""Callable task entry points, loaded only when their task is used."""
from importlib import import_module

from .task_config import (
    ArrayConfig, IntensitiesConfig, SnvConfig, ScanConfig, MotifsConfig,
    CalibrationConfig,
)

TASKS = {
    'array': ('array', 'run_array', ArrayConfig),
    'intensities': ('intensities', 'run_intensities', IntensitiesConfig),
    'snv': ('snv', 'run_snv', SnvConfig),
    'scan': ('scan', 'run_scan', ScanConfig),
    'motifs': ('motifs', 'run_motifs', MotifsConfig),
    'calibrate_max_probes': ('calibrate', 'run_max_probe_calibration', CalibrationConfig),
    'calibrate_mask': ('calibrate', 'run_mask_calibration', CalibrationConfig),
}


def task_runner(task):
    module, name, _ = TASKS[task]
    return getattr(import_module(f'eubar.{module}'), name)
