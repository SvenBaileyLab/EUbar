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
    # from utils import read_kmer_positions
    import tempfile

    test_data = "CTGAACTT\tchr14:23094440-23096032;2;1433 1569,\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(test_data)
        tmp_path = tmp.name

    result = read_kmer_positions(tmp_path)

    assert result["CTGAACTT"]["chr14:23094440-23096032"] == 1433
    assert result["CTGAACTT"]["chr14:23094440-23096032.1"] == 1569


def test_get_kmer_variants():
    motif = "GGGCAAACCAGGTAA"  # 15-mer
    k = 8
    kmers, variants, snp_indices = get_kmer_variants(motif, k)

    assert len(kmers) == 8  # 15 - 8 + 1
    assert variants[0] == "GGGCAAA."
    assert snp_indices[0] == 7
    assert "." in variants[0]


def test_match_kmers_to_wildcards():
    motif = "GGGCAAACCAGGTAA"
    k = 8
    _, variants, snp_indices = get_kmer_variants(motif, k)

    # Variant[0] = "GGGCAAA.", snp_index = 7, ref_allele = motif[0+7] = "A"
    kmer_positions = {
        "GGGCAAAA": {"chr1:100-200": 42},  # matches the wildcard
        "GGGCAAAC": {"chr1:300-400": 99},  # also matches
        "TTTTTTTT": {"chr1:999-1111": 55},  # doesn't match anything
    }

    comp_alleles, matched = match_kmers_to_wildcards(
        kmer_positions, variants, snp_indices, motif
    )

    assert 0 in matched
    assert "A" in matched[0]
    assert "C" in matched[0]
    assert matched[0]["A"]["chr1:100-200"] == 42
    assert matched[0]["C"]["chr1:300-400"] == 99
    assert snp_indices[0] == 7
    assert motif[0 + snp_indices[0]] == "C"


def test_select_random_probes():
    kmer_positions = {
        "AAA": {"region1": 10, "region2": 20},
        "CCC": {"region3": 30, "region4": 40},
        "GGG": {"region5": 50},
    }

    matched = {"region2", "region3"}  # Exclude these

    result = select_random_probes(kmer_positions, matched, num_random=2, seed=123)

    # Make sure no excluded regions are in result
    for region in result:
        assert region not in matched

    # Check that we got the right number of probes
    assert len(result) == 2

    # Check that selected regions are from the expected pool
    valid = {"region1", "region4", "region5"}
    for region in result:
        assert region in valid


def test_build_allele_matrix_comprehensive():
    matched = {
        0: {
            7: {
                "A": {"region1": 0, "region2": 5, "region4": 9},
                "C": {"region3": 2, "region4": 9},  # region4 appears twice — should only be used once
                "T": {"region5": 6},
                "G": {"region_missing": 10},  # this one is not in probe_intensities
            }
        }
    }

    probe_intensities = {
        "region1": 5.0,
        "region2": 6.0,
        "region3": 7.0,
        "region4": 8.0,
        "region5": 9.0,
        # region_missing is intentionally omitted
    }

    matrix, values = build_allele_matrix(matched, probe_intensities)

    # Expected order of seen regions: region1, region2, region3, region4, region5
    assert values == [5.0, 6.0, 8.0, 7.0, 9.0]  

    # Matrix should have 5 rows (one per included region) and 4 columns (one per allele)
    assert matrix["A"] == [1, 1, 1, 0, 0]
    assert matrix["C"] == [0, 0, 0, 1, 0]
    assert matrix["T"] == [0, 0, 0, 0, 1]
    assert matrix["G"] == [0, 0, 0, 0, 0]

    # Check that each row has exactly one 1
    for i in range(len(values)):
        total = matrix["A"][i] + matrix["C"][i] + matrix["T"][i] + matrix["G"][i]
        assert total == 1

def test_build_allele_matrix_exclude_ref():
    matched = {
        0: {
            7: {
                "A": {"region1": 0, "region2": 1},
                "C": {"region3": 2},
                "T": {"region4": 3},
            }
        }
    }

    probe_intensities = {
        "region1": 5.0,
        "region2": 6.0,
        "region3": 7.0,
        "region4": 8.0,
    }

    # Exclude "A" from the matrix
    matrix, values = build_allele_matrix(matched, probe_intensities, exclude_alleles={"A"})

    # We should have only seen r3 and r4
    assert values == [7.0, 8.0]

    # Only C and T should appear
    assert set(matrix.keys()) == {"C", "T"}

    assert matrix["C"] == [1, 0]
    assert matrix["T"] == [0, 1]


def test_run_regression_basic():
    # Simulated binary allele matrix
    allele_matrix = {
        "A": [1, 0, 1, 0],
        "C": [0, 1, 0, 1],
    }
    probe_values = [10.0, 15.0, 12.0, 18.0]
    extra_covars = {"lp": [0.5, 0.5, 0.5, 0.5]}

    stats = run_regression(allele_matrix, probe_values, mode="neg-binomial", extra_covariates=extra_covars)

    assert "A" in stats
    assert "C" in stats
    assert "lp" in stats

    for coef, stderr, pval in stats.values():
        assert isinstance(coef, float)
        assert isinstance(stderr, float)
        assert isinstance(pval, float)


def _test_run_aff_rand_regression_excludes_ref_aff():
    matched = {
        0: {
            7: {
                "A": {"chr1:100-200": 0, "chr1:201-300": 0},
                "C": {"chr1:301-400": 0},
                "G": {"chr1:401-500": 0},
            }
        }
    }

    random = {
        0: {
            7: {
                "A": {"chr1:501-600": 0},
                "C": {"chr1:601-700": 0},
                "G": {"chr1:701-800": 0},
                "T": {"chr1:801-900": 0},
            }
        }
    }

    intensities = {
        "chr1:100-200": 10.0,
        "chr1:201-300": 12.0,
        "chr1:301-400": 15.0,
        "chr1:401-500": 13.0,
        "chr1:501-600": 11.0,
        "chr1:601-700": 14.0,
        "chr1:701-800": 9.0,
        "chr1:801-900": 16.0,
    }

    comp_alleles = {
        0: {7: "A"}
    }

    aff_stats, rand_stats = run_aff_rand_regression(
        matched, random, intensities,
        comp_alleles=comp_alleles,
        dhs=False,
        mode="log-linear"
    )

    # AFF should exclude "A"
    assert "A" not in aff_stats
    assert "C" in aff_stats
    assert "G" in aff_stats

    # RAND should include all
    for allele in "ACGT":
        assert allele in rand_stats
