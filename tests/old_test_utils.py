import sys
import os
import warnings
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    read_intensities,
    read_kmer_positions,
    get_kmer_variants,
    match_kmers_to_wildcards,
    select_random_probes,
    build_allele_matrix,
    run_regression,
    run_aff_rand_regression,
    print_results
)


def test_read_intensities():
    result = read_intensities("tests/data/test_intensities.txt")
    assert result["chr1:842880-843060"] == 5.69493


def test_read_kmer_positions():
    result = read_kmer_positions("tests/data/test_kmer_positions.txt")
    assert result["CCATGAAG"]["chr1:778660-778800"] == 100


def test_read_kmer_positions_multi():
    from utils import read_kmer_positions
    import tempfile

    test_data = "CTGAACTT\tchr14:23094440-23096032;2;1433 1569,\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(test_data)
        tmp_path = tmp.name

    result = read_kmer_positions(tmp_path)

    assert result["CTGAACTT"]["chr14:23094440-23096032"] == 1433
    assert result["CTGAACTT"]["chr14:23094440-23096032.1"] == 1569


def test_get_kmer_variants():
    motif = "CGGGTTTTGGCAGTA"
    kmer_size = 8
    kmers, variants = get_kmer_variants(motif, kmer_size)

    assert len(kmers) == len(motif) - kmer_size + 1  # 8
    assert kmers[0] == "CGGGTTTT"
    assert kmers[-1] == "TGGCAGTA"
    assert (0, 0) in variants
    assert (0, 7) in variants
    assert variants[(0, 0)] == ".GGGTTTT"
    assert variants[(0, 7)] == "CGGGTTT."


def test_match_kmers_to_wildcards():
    kmer_positions = {
        "AGGGTTTT": {"chr1:100-200": 42},
        "CGGGTTTT": {"chr1:300-400": 50},
        "TGGGTTTT": {"chr1:500-600": 80},
    }

    wildcard_variants = {
        (0, 0): ".GGGTTTT",
    }

    comp_alleles, matched = match_kmers_to_wildcards(kmer_positions, wildcard_variants)

    assert 0 in matched
    assert 0 in matched[0]

    alleles = matched[0][0]
    assert "A" in alleles
    assert alleles["A"]["chr1:100-200"] == 42
    assert alleles["C"]["chr1:300-400"] == 50
    assert alleles["T"]["chr1:500-600"] == 80

    assert comp_alleles[0][0] in ["A", "C", "T"]


def test_select_random_probes():
    kmer_positions = {
        "AAAAAAA": {"chr1:100-200": 10, "chr1:300-400": 20, "chr1:500-600": 30},
        "CCCCCCC": {"chr2:700-800": 40, "chr2:900-1000": 50, "chr3:1100-1200": 60},
    }

    matched_probes = {"chr1:300-400", "chr2:900-1000"}

    result = select_random_probes(kmer_positions, matched_probes, num_random=2, seed=1)

    # Ensure only unmatched probes are in result
    for region in result:
        assert region not in matched_probes

    assert len(result) == 2

def test_build_allele_matrix():
    matched = {
        0: {
            0: {
                'A': {'chr1:100-200': 42},
                'C': {'chr1:300-400': 50}
            }
        }
    }

    probe_intensities = {
        'chr1:100-200': 2.5,
        'chr1:300-400': 3.5
    }

    matrix, values = build_allele_matrix(matched, probe_intensities)

    assert matrix['A'] == [1, 0]
    assert matrix['C'] == [0, 1]
    assert values == [2.5, 3.5]

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_regression_log_linear():
    allele_matrix = {
        "A": [1, 0, 0, 1],
        "C": [0, 1, 0, 0],
        "T": [0, 0, 1, 0]
    }
    probe_values = [5.0, 2.5, 3.0, 5.5]
    extra_covariates = {
        "lp": [0.2, 0.4, 0.3, 0.1]
    }

    stats = run_regression(allele_matrix, probe_values, mode="log-linear", extra_covariates=extra_covariates)

    assert "A" in stats
    assert "lp" in stats
    assert isinstance(stats["lp"], tuple)
    assert len(stats["lp"]) == 3  # coef, stderr, pval

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_regression_log_linear_dhs():
    allele_matrix = {
        "A": [1, 0, 1, 0],
        "C": [0, 1, 0, 1]
    }
    probe_values = [6.0, 2.0, 3.5, 4.5]
    extra_covariates = {
        "lp": [0.3, 0.2, 0.5, 0.1],
        "sl": [150, 120, 130, 110]
    }

    stats = run_regression(allele_matrix, probe_values, mode="log-linear", extra_covariates=extra_covariates)

    assert "A" in stats
    assert "C" in stats
    assert "lp" in stats
    assert "sl" in stats
    assert isinstance(stats["sl"], tuple)

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_aff_rand_regression_basic():
    matched = {
        0: {
            0: {
                "A": {"chr1:100-200": 40, "chr1:300-400": 50},
                "C": {"chr1:500-600": 80}
            }
        }
    }
    random = {
        0: {
            0: {
                "A": {"chr1:700-800": 30},
                "C": {"chr1:900-1000": 60}
            }
        }
    }
    probe_intensities = {
        "chr1:100-200": 4.0,
        "chr1:300-400": 5.0,
        "chr1:500-600": 6.0,
        "chr1:700-800": 3.0,
        "chr1:900-1000": 2.5
    }

    aff_stats, rand_stats = run_aff_rand_regression(matched, random, probe_intensities, dhs=False, mode="log-linear")

    assert "A" in aff_stats
    assert "C" in aff_stats
    assert "lp" in aff_stats
    assert isinstance(aff_stats["lp"], tuple)
    assert len(aff_stats["lp"]) == 3

    assert "A" in rand_stats
    assert "C" in rand_stats
    assert "lp" in rand_stats

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_aff_rand_regression_dhs():
    matched = {
        0: {
            0: {
                "A": {"chr1:100-200": 40, "chr1:300-400": 50},
                "T": {"chr1:500-600": 80}
            }
        }
    }
    random = {
        0: {
            0: {
                "A": {"chr1:700-800": 30},
                "T": {"chr1:900-1000": 60}
            }
        }
    }
    probe_intensities = {
        "chr1:100-200": 4.0,
        "chr1:300-400": 5.0,
        "chr1:500-600": 6.0,
        "chr1:700-800": 3.0,
        "chr1:900-1000": 2.5
    }

    aff_stats, rand_stats = run_aff_rand_regression(matched, random, probe_intensities, dhs=True, mode="log-linear")

    assert "sl" in aff_stats
    assert "sl" in rand_stats
    assert isinstance(aff_stats["sl"], tuple)
    assert isinstance(rand_stats["sl"], tuple)

def test_print_results(capfd):
    results = {
        "AFF": {
            0: {
                0: {
                    "A": (0.12, 0.03, 0.001),
                    "T": (-0.05, 0.02, 0.045),
                    "C": (0.0, 0.01, float("nan"))
                }
            }
        },
        "RAND": {
            0: {
                0: {
                    "A": (0.02, 0.04, 0.6),
                    "T": (-0.03, 0.01, 0.2)
                }
            }
        }
    }

    mutation_id = "chr6:41071106:C>T"
    print_results(results, mutation_id)

    out, _ = capfd.readouterr()
    lines = out.strip().split("\n")

    # Check AFF lines
    assert "AFF\tchr6:41071106:C>T\tA\t0\t0.12000000\t0.00100000" in lines
    assert "AFF\tchr6:41071106:C>T\tT\t0\t-0.05000000\t0.04500000" in lines
    assert "AFF\tchr6:41071106:C>T\tC\t0\t0.00000000\tNA" in lines

    # Check RAND lines
    assert "RAND\tchr6:41071106:C>T\tA\t0\t0.02000000\t0.60000000" in lines
    assert "RAND\tchr6:41071106:C>T\tT\t0\t-0.03000000\t0.20000000" in lines
