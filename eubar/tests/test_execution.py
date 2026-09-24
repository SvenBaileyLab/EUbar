"""Outputs recorded from e47fda5 before the structural refactor."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from execution_cases import CASES, FIXTURES, arguments
from output_comparison import assert_output_equal, assert_table_equal

ROOT = Path(__file__).resolve().parents[2]


def run_cli(args):
    return subprocess.run(
        [sys.executable, '-m', 'eubar', *args], cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT), 'MPLBACKEND': 'Agg',
             'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True,
        timeout=120,
    )


@pytest.mark.parametrize('name', CASES)
def test_recorded_outputs(name, tmp_path):
    result = run_cli(arguments(name, tmp_path))
    assert result.returncode == 0, result.stderr
    expected = FIXTURES / 'expected' / name
    if name.startswith(('snv', 'scan')):
        assert_table_equal(result.stdout, (expected / 'stdout.txt').read_text())
        assert len(result.stdout.splitlines()) > 1
    else:
        files = sorted(p.name for p in expected.iterdir() if p.suffix in {'.tsv', '.meme'})
        assert files
        assert sorted(p.name for p in tmp_path.iterdir() if p.suffix in {'.tsv', '.meme'}) == files
        for filename in files:
            assert_output_equal(tmp_path / filename, expected / filename)
        if name.startswith('motifs'):
            from PIL import Image
            plots = list(tmp_path.glob('*.png'))
            assert len(plots) >= 4
            for path in plots:
                with Image.open(path) as img:
                    img.verify()


@pytest.mark.parametrize('name', ['snv_best_holm_cap', 'snv_mask_best_holm_cap'])
def test_parallel_output_matches_serial(name, tmp_path):
    args = arguments(name, tmp_path)
    serial = run_cli(args)
    for _ in range(3):
        parallel = run_cli([*args, '--jobs', '2'])
        assert serial.returncode == parallel.returncode == 0, parallel.stderr
        assert serial.stdout == parallel.stdout


def test_array_output_and_legacy_aliases(tmp_path):
    for flags in [('--kmer-size', '--output'), ('--kmer_size', '--out')]:
        out = tmp_path / 'array.tsv'
        result = run_cli(['array', '--bed', str(FIXTURES/'probes.bed'), '--genome',
                          str(FIXTURES/'genome.fa'), flags[0], '4', flags[1], str(out)])
        assert result.returncode == 0, result.stderr
        assert out.read_bytes() == (FIXTURES/'array.tsv').read_bytes()


def test_intensity_default_output(tmp_path):
    import shutil
    if not shutil.which('bedtools'):
        pytest.skip('bedtools is required for intensity extraction')
    out = tmp_path / 'signal.tsv'
    result = run_cli(['intensities', '--bed', str(FIXTURES/'probes.bed'),
                      '--signal', str(FIXTURES/'signal.bw'), '--genome-fasta',
                      str(FIXTURES/'genome.fa'), '--output', str(out)])
    assert result.returncode == 0, result.stderr
    assert_output_equal(out, FIXTURES/'expected/intensities.tsv')


@pytest.mark.parametrize('name', ['snv_best_holm_cap', 'snv_mask_best_holm_cap'])
def test_spawn_workers_match_reference(name, tmp_path):
    script = tmp_path/'spawn_job.py'
    script.write_text(
        "import multiprocessing as mp\nimport sys\nfrom eubar import snv\n"
        "if __name__ == '__main__':\n"
        "    snv._preferred_mp_context = lambda: mp.get_context('spawn')\n"
        "    raise SystemExit(snv.main(sys.argv[1:]))\n"
    )
    args = arguments(name, tmp_path)[1:] + ['--jobs', '2']
    result = subprocess.run([sys.executable, str(script), *args], cwd=ROOT,
                            env={**os.environ, 'PYTHONPATH':str(ROOT), 'OPENBLAS_NUM_THREADS':'1'},
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert_table_equal(result.stdout, (FIXTURES/'expected'/name/'stdout.txt').read_text())
