from __future__ import annotations

from eubar.calibrate import build_parser


def test_calibrate_help_hides_legacy_aliases_and_internal_thresholds():
    help_text = build_parser().format_help()
    for hidden in (
        "--kmerPositions",
        "--out ",
        "--genome-fasta",
        "--n-kmers",
        "--candidate-multiplier",
        "--max-effect-sd",
        "--max-effect-bias",
        "--min-eligible-fraction",
        "--borderline-relative-tolerance",
        "--near-optimal-relative-tolerance",
    ):
        assert hidden not in help_text
    for public in ("--array", "--out-prefix", "--genome", "--n-families"):
        assert public in help_text


def test_calibrate_legacy_aliases_still_parse():
    args = build_parser().parse_args([
        "mask",
        "--intensities", "tf.int",
        "--kmerPositions", "array.txt",
        "--out", "result",
        "--genome-fasta", "hg38.fa",
        "--n-kmers", "123",
    ])
    assert args.array == "array.txt"
    assert args.out_prefix == "result"
    assert args.genome == "hg38.fa"
    assert args.n_families == 123
