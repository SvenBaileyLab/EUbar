import argparse
from pybedtools import BedTool
import os
import tempfile
import pyBigWig
import math
import sys
import numpy as np


BIGWIG_EXTS = (".bw", ".bigwig", ".bigWig", ".BW")
SUMMARY_CHOICES = ("max", "mean", "center_max", "center_mean")


def deduplicate_bed_by_coords(bed_path):
    temp_bed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".bed").name
    seen = set()
    with open(bed_path) as infile, open(temp_bed_path, "w") as outfile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            key = tuple(fields[:3])  # chrom, start, end
            if key not in seen:
                seen.add(key)
                outfile.write(line)
    print(f"[Info] Deduplicated BED written to temporary file: {temp_bed_path}")
    return temp_bed_path


def is_bigwig(path: str) -> bool:
    return path.endswith(BIGWIG_EXTS)


def safe_stat(bw, chrom, start, end, stat: str):
    """
    Returns BigWig summary in [start, end) using `stat` in {"max","mean"}.
    If missing/None/NaN, returns 0.0.
    """
    try:
        v = bw.stats(chrom, start, end, type=stat)
        if not v:
            return 0.0
        val = v[0]
        if val is None:
            return 0.0
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return 0.0
        return float(val)
    except RuntimeError:
        # e.g., chrom not in bigwig
        return 0.0


def _clamp_interval(start: int, end: int, chrom_len: int):
    s = max(0, min(start, chrom_len))
    e = max(0, min(end, chrom_len))
    if e <= s:
        return None
    return s, e


def _center_window(start: int, end: int, window_bp: int):
    mid = (start + end) // 2
    half = window_bp // 2
    s = mid - half
    e = s + window_bp  # ensures exact window_bp length when possible
    return s, e


def run_bigwig_summary_per_interval(
    a_bed,
    bigwig_path,
    output_path,
    summary="max",
    window_bp=100,
    genome=None,
    keep_temp=False,
):
    if summary not in SUMMARY_CHOICES:
        raise ValueError(f"summary must be one of {SUMMARY_CHOICES}, got {summary}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dedup_a_bed = deduplicate_bed_by_coords(a_bed)

    # Sort A for consistent output order (not required for BigWig querying)
    a = BedTool(dedup_a_bed)
    a = a.sort(g=genome) if genome else a.sort()

    # Map our summary modes to pyBigWig's stats
    if summary in ("max", "center_max"):
        bw_stat = "max"
    else:
        bw_stat = "mean"

    with pyBigWig.open(bigwig_path) as bw, open(output_path, "w") as out:
        bw_chroms = bw.chroms()  # dict of chrom -> length

        for interval in a:
            chrom = interval.chrom
            start = int(interval.start)
            end = int(interval.end)

            if chrom not in bw_chroms:
                signal = 0.0
            else:
                clen = int(bw_chroms[chrom])

                if summary.startswith("center_"):
                    s0, e0 = _center_window(start, end, window_bp)
                else:
                    s0, e0 = start, end

                clamped = _clamp_interval(s0, e0, clen)
                if clamped is None:
                    signal = 0.0
                else:
                    s, e = clamped
                    signal = safe_stat(bw, chrom, s, e, stat=bw_stat)

            # IMPORTANT: keep the output key as the ORIGINAL DNase/ATAC interval
            out.write(f"{chrom}:{start}-{end}\t{signal}\n")

    print(f"[Done] Output written to: {output_path}")

    if not keep_temp:
        os.remove(dedup_a_bed)
        print(f"[Cleanup] Removed temporary deduplicated BED: {dedup_a_bed}")


def _write_center_window_bed(sorted_a_bedtool: BedTool, window_bp: int):
    """
    Create a temp BED where each interval is replaced by a fixed center window.
    We keep chrom and new start/end; downstream we still output ORIGINAL keys
    (so this is only for bedtools map coordinates).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bed").name
    with open(tmp, "w") as out:
        for iv in sorted_a_bedtool:
            chrom = iv.chrom
            start = int(iv.start)
            end = int(iv.end)
            s0, e0 = _center_window(start, end, window_bp)
            # bedtools can handle negative starts poorly; clamp to >=0
            s0 = max(0, s0)
            e0 = max(s0 + 1, e0)  # ensure at least 1bp
            out.write(f"{chrom}\t{s0}\t{e0}\n")
    return tmp


def run_bedtools_map_pybedtools(
    a_bed,
    b_bedgraph,
    output_path,
    summary="max",
    window_bp=100,
    genome=None,
    keep_temp=False,
):
    """
    BedGraph path (non-BigWig). We sort BOTH A and B to avoid bedtools map sort errors.
    Supports summary in {max, mean, center_max, center_mean}.
    """
    if summary not in SUMMARY_CHOICES:
        raise ValueError(f"summary must be one of {SUMMARY_CHOICES}, got {summary}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dedup_a_bed = deduplicate_bed_by_coords(a_bed)

    a = BedTool(dedup_a_bed)
    b = BedTool(b_bedgraph)

    a_sorted = a.sort(g=genome) if genome else a.sort()
    b_sorted = b.sort(g=genome) if genome else b.sort()

    # For center_* summaries, create a windowed-A for mapping
    tmp_center_bed = None
    if summary.startswith("center_"):
        tmp_center_bed = _write_center_window_bed(a_sorted, window_bp=window_bp)
        a_map = BedTool(tmp_center_bed)
        a_map = a_map.sort(g=genome) if genome else a_map.sort()
        map_op = "max" if summary == "center_max" else "mean"
    else:
        a_map = a_sorted
        map_op = "max" if summary == "max" else "mean"

    # bedtools map: map bedgraph col4 onto intervals
    result = a_map.map(b=b_sorted, c=4, o=map_op)

    # IMPORTANT: output keys should still correspond to ORIGINAL DNase/ATAC intervals.
    # Since `result` is based on a_map (which may be center windows), we can't trust
    # result's start/end for the key. We'll iterate ORIGINAL sorted A alongside results.
    with open(output_path, "w") as out:
        # Need a re-iterable list of original sorted intervals
        orig_intervals = list(a_sorted)

        # result is iterable in same order/length as a_map; a_map has same number of lines as a_sorted
        for orig_iv, mapped_iv in zip(orig_intervals, result):
            chrom = orig_iv.chrom
            start = int(orig_iv.start)
            end = int(orig_iv.end)

            fields = mapped_iv.fields
            val = fields[-1] if fields[-1] != "." else "0"
            out.write(f"{chrom}:{start}-{end}\t{val}\n")

    print(f"[Done] Output written to: {output_path}")

    if not keep_temp:
        os.remove(dedup_a_bed)
        print(f"[Cleanup] Removed temporary deduplicated BED: {dedup_a_bed}")
        if tmp_center_bed is not None:
            os.remove(tmp_center_bed)
            print(f"[Cleanup] Removed temporary center-window BED: {tmp_center_bed}")


# ---------------- Residualization helpers ----------------

_rc = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def _parse_region_key(region: str):
    """Parse 'chr:start-end' -> (chr, start, end)."""
    chrom, rest = region.split(":", 1)
    start_s, end_s = rest.split("-", 1)
    return chrom, int(start_s), int(end_s)

def _make_fasta_fetcher(genome_fa: str):
    """Return fetch(chrom, start, end) -> uppercase sequence."""
    try:
        import pysam  # type: ignore
        fa = pysam.FastaFile(genome_fa)
        def fetch(chrom: str, start: int, end: int) -> str:
            return fa.fetch(chrom, start, end).upper()
        return fetch
    except Exception:
        try:
            from pyfaidx import Fasta  # type: ignore
            fa = Fasta(genome_fa, as_raw=True, sequence_always_upper=True)
            def fetch(chrom: str, start: int, end: int) -> str:
                return str(fa[chrom][start:end]).upper()
            return fetch
        except Exception as e:
            raise RuntimeError(
                "Could not open genome FASTA. Install pysam or pyfaidx, and check path. " 
                f"Error: {e}"
            )

def _gc_fraction(seq: str) -> float:
    seq = seq.upper()
    # count only A/C/G/T (ignore N)
    a = seq.count("A"); c = seq.count("C"); g = seq.count("G"); t = seq.count("T")
    denom = a + c + g + t
    if denom == 0:
        return 0.0
    return float(c + g) / float(denom)

def _load_openness_from_bed(path: str):
    """
    Try to load an 'openness' covariate from the DNase/ATAC regions file itself.
    Supports common peak formats (8-col custom, narrowPeak, BED with numeric score).
    Returns: (open_map, used_col_name_or_guess, n_missing)
    where open_map maps region_key 'chr:start-end' -> open_value.
    """
    # Read first non-comment, non-empty line to infer ncols
    first = None
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            first = line.rstrip("\n")
            break
    if first is None:
        return {}, None, 0

    ncols = len(first.split("\t"))
    open_map = {}
    used = None

    # Heuristic: choose an 'open' column index based on known formats,
    # otherwise choose the numeric column (>=col4) with best parse rate.
    fixed_idx = None
    fixed_name = None
    if ncols == 8:
        fixed_idx, fixed_name = 7, "smoothed_peak_height"
    elif ncols >= 10:
        fixed_idx, fixed_name = 6, "signalValue"  # narrowPeak col7 (0-based 6)

    # First pass: collect values
    numeric_cols = [0] * ncols
    numeric_ok = [0] * ncols

    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            a = line.rstrip("\n").split("\t")
            if len(a) < 3:
                continue
            chrom = a[0]
            try:
                start = int(a[1]); end = int(a[2])
            except ValueError:
                continue
            region = f"{chrom}:{start}-{end}"
            rows.append((region, a))
            for j in range(3, len(a)):
                numeric_cols[j] += 1
                try:
                    float(a[j])
                    numeric_ok[j] += 1
                except Exception:
                    pass

    if not rows:
        return {}, None, 0

    if fixed_idx is not None and fixed_idx < ncols:
        chosen_idx = fixed_idx
        used = fixed_name
    else:
        # choose numeric column with highest parse success rate
        best_j = None
        best_rate = -1.0
        for j in range(3, ncols):
            if numeric_cols[j] == 0:
                continue
            rate = numeric_ok[j] / numeric_cols[j]
            if rate > best_rate:
                best_rate = rate
                best_j = j
        if best_j is None or best_rate < 0.80:
            # no reliable numeric column
            return {}, None, 0
        chosen_idx = best_j
        used = f"col{chosen_idx+1}"

    missing = 0
    for region, a in rows:
        if chosen_idx >= len(a):
            missing += 1
            continue
        try:
            open_map[region] = float(a[chosen_idx])
        except Exception:
            missing += 1

    return open_map, used, missing

def _read_two_col_map(path: str):
    """Read 'key value' lines into dict."""
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            out[key] = val
    return out

def residualize_intensities(int_map, bed_path: str, genome_fa: str):
    """
    Fit: log1p(signal) ~ 1 + gc + length [+ log1p(open) if available for ALL regions]
    Return: residual_map (same keys), plus a dict of coefficients and the formula string.
    Prints a coefficient table to stdout.
    """
    fetch = _make_fasta_fetcher(genome_fa)

    regions = list(int_map.keys())

    # Covariates: gc + length
    gc = np.zeros(len(regions), dtype=float)
    length = np.zeros(len(regions), dtype=float)
    y = np.zeros(len(regions), dtype=float)

    for i, r in enumerate(regions):
        chrom, start, end = _parse_region_key(r)
        seq = fetch(chrom, start, end)
        gc[i] = _gc_fraction(seq)
        length[i] = float(end - start)
        yv = float(int_map[r])
        y[i] = yv

    y_log = np.log1p(np.maximum(y, 0.0))

    # Openness from the BED itself (optional, only if complete coverage)
    open_map, open_col, open_missing = _load_openness_from_bed(bed_path)
    use_open = False
    open_vec = None
    if open_map and open_missing == 0:
        tmp = np.array([open_map.get(r, np.nan) for r in regions], dtype=float)
        if np.isnan(tmp).sum() == 0:
            use_open = True
            open_vec = np.log1p(np.maximum(tmp, 0.0))

    # Build design matrix
    cols = [np.ones(len(regions), dtype=float), gc, length]
    names = ["intercept", "gc", "length"]
    formula = "y_log ~ 1 + gc + length"
    if use_open:
        cols.append(open_vec)
        names.append("log1p(open)")
        formula = "y_log ~ 1 + gc + length + log1p(open)"

    X = np.vstack(cols).T  # n x p

    # Fit with least squares
    beta, *_ = np.linalg.lstsq(X, y_log, rcond=None)
    yhat = X @ beta
    resid = y_log - yhat

    # Print coefficient table
    print(f"Coefficients ({formula}):")
    for n, b in zip(names, beta):
        print(f"  {n}: {float(b)}")
    if use_open:
        print(f"[Info] openness covariate loaded from --bed using column: {open_col} (coverage: 100%)")
    else:
        if open_map:
            # had a column but incomplete or messy
            miss = int(np.isnan(np.array([open_map.get(r, np.nan) for r in regions], dtype=float)).sum())
            print(f"[Info] openness covariate NOT used (missing for {miss}/{len(regions)} regions).")
        else:
            print("[Info] openness covariate NOT used (no numeric openness column detected in --bed).")

    resid_map = {r: float(resid[i]) for i, r in enumerate(regions)}
    coef_map = {names[i]: float(beta[i]) for i in range(len(names))}
    return resid_map, coef_map, formula


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ChIP signal over DNase/ATAC regions from either BigWig or BedGraph.\n"
            "Outputs: region_key<TAB>signal, where region_key is chrom:start-end of the ORIGINAL regions."
        )
    )
    parser.add_argument("--bed", required=True, help="Path to DNase-seq/ATAC-seq BED file (-a)")
    parser.add_argument("--signal", required=True, help="Path to ChIP-seq BedGraph OR BigWig file (-b)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--genome", default=None, help="Optional genome chrom sizes file for bedtools sort (-g)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")

    parser.add_argument(
        "--residualize",
        action="store_true",
        help=(
            "If set, fit a simple background model and output residualized intensities. "
            "Model: log1p(signal) ~ 1 + GC + length (+ optional log1p(open) if available). "
            "Requires --genome-fasta. Coefficients are printed to stdout."
        ),
    )
    parser.add_argument(
        "--genome-fasta",
        default=None,
        help="Genome FASTA (required when --residualize is set).",
    )

    # NEW:
    parser.add_argument(
        "--summary",
        default="max",  # keep your current behavior
        choices=SUMMARY_CHOICES,
        help="How to summarize signal: max/mean over full interval, or center_max/center_mean over a fixed window.",
    )
    parser.add_argument(
        "--window-bp",
        type=int,
        default=100,
        help="Window size (bp) for center_* summaries (default: 100). Ignored for max/mean.",
    )

    args = parser.parse_args(argv)


    if args.residualize and not args.genome_fasta:
        parser.error("--genome-fasta is required when --residualize is set.")

    # If residualizing, write raw extracted signal to a temp file first,
    # then overwrite --output with residualized values.
    tmp_out = args.output
    tmp_created = False
    if args.residualize:
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".signal.txt").name
        tmp_created = True

    if is_bigwig(args.signal):
        run_bigwig_summary_per_interval(
            args.bed,
            args.signal,
            tmp_out,
            summary=args.summary,
            window_bp=args.window_bp,
            genome=args.genome,
            keep_temp=args.keep_temp,
        )
    else:
        run_bedtools_map_pybedtools(
            args.bed,
            args.signal,
            tmp_out,
            summary=args.summary,
            window_bp=args.window_bp,
            genome=args.genome,
            keep_temp=args.keep_temp,
        )

    if args.residualize:
        sig_map = _read_two_col_map(tmp_out)
        resid_map, _, _ = residualize_intensities(sig_map, bed_path=args.bed, genome_fa=args.genome_fasta)

        # Write residualized map to the requested output
        outdir = os.path.dirname(args.output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.output, "w") as out:
            for k, v in resid_map.items():
                out.write(f"{k}\t{v}\n")
        print(f"[Done] Residualized output written to: {args.output}")

        if (not args.keep_temp) and tmp_created:
            try:
                os.remove(tmp_out)
            except OSError:
                pass

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
