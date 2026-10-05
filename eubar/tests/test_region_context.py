"""Regional CpG o/e covariate: the ratio itself, its cache, and the --no-cpg-covariate switch."""
import math
import shutil

import pandas as pd
import pytest
from execution_cases import CASES, FIXTURES, arguments
from output_comparison import assert_table_equal
from test_execution import run_cli

from eubar.core.design import DesignBuilder
from eubar.core.region_context import cpg_oe, default_cache_path, load_region_cpg_oe

CPG_CASES = [c for c in CASES if c.startswith(("snv", "scan")) and c != "scan_pooled"]


def test_cpg_oe_formula():
    # ACGTACGTCC: 4 C, 2 G, 2 CG in 10 bp -> 2 * 10 / (4 * 2)
    assert cpg_oe("ACGTACGTCC") == pytest.approx(2 * 10 / (4 * 2))
    assert cpg_oe("cgcg") == pytest.approx(2 * 4 / (2 * 2))       # case-insensitive
    assert cpg_oe("AAAATTTT") == 0.0                                 # no C or no G
    assert cpg_oe("CCCCAAAA") == 0.0


def test_load_uses_and_extends_cache(tmp_path):
    genome = tmp_path / "genome.fa"
    shutil.copy(FIXTURES / "genome.fa", genome)
    regions = list(pd.read_csv(FIXTURES / "intensities.tsv", sep=r"\s+", header=None)[0][:20])
    cache = tmp_path / "x.cpg_oe.tsv"
    first = load_region_cpg_oe(regions[:10], str(genome), cache_path=str(cache))
    assert cache.is_file() and len(cache.read_text().splitlines()) == 10
    both = load_region_cpg_oe(regions, str(genome), cache_path=str(cache))
    assert len(cache.read_text().splitlines()) == 20
    assert all(both[r] == first[r] for r in regions[:10])
    assert not list(tmp_path.glob("*.tmp"))                         # atomic write leaves nothing behind


def test_unreadable_region_gets_median_in_design():
    vals = {"chrNope:0-10": 1.0}
    cpg = load_region_cpg_oe(vals, str(FIXTURES / "genome.fa"), cache_path=None)
    assert math.isnan(cpg["chrNope:0-10"])
    lookup = {"A": {f"chr1:{i}-{i + 20}": 3 for i in range(0, 120, 10)},
              "C": {f"chr1:{i}-{i + 20}": 3 for i in range(300, 420, 10)}}
    ints = {r: 0.0 for d in lookup.values() for r in d}
    cpg = {r: (float("nan") if i == 0 else 0.5) for i, r in enumerate(ints)}
    wd = DesignBuilder(ints, region_cpg_oe=cpg).build(region_lookup=lookup, ref_allele="A")
    assert wd.X["cpg_oe"].notna().all() and wd.X["cpg_oe"].iloc[0] == 0.5


def test_no_covariate_columns_without_cpg():
    lookup = {"A": {f"chr1:{i}-{i + 20}": 3 for i in range(0, 120, 10)},
              "C": {f"chr1:{i}-{i + 20}": 3 for i in range(300, 420, 10)}}
    ints = {r: 0.0 for d in lookup.values() for r in d}
    wd = DesignBuilder(ints).build(region_lookup=lookup, ref_allele="A")
    assert "cpg_oe" not in wd.X.columns


def test_default_cache_path():
    assert default_cache_path("/x/y.tsv") == "/x/y.tsv.cpg_oe.tsv"


@pytest.mark.parametrize("name", CPG_CASES)
def test_no_cpg_covariate_reproduces_previous_release(name, tmp_path):
    """With --no-cpg-covariate, SNV and scan output is byte-identical to the pre-CpG recordings."""
    result = run_cli(arguments(name, tmp_path) + ["--no-cpg-covariate"])
    assert result.returncode == 0, result.stderr
    assert_table_equal(result.stdout, (FIXTURES / "expected" / name / "stdout_no_cpg.txt").read_text())
