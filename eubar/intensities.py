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
    genome_size_file=None,
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
    a = a.sort(g=genome_size_file) if genome_size_file else a.sort()

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
    genome_size_file=None,
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

    a_sorted = a.sort(g=genome_size_file) if genome_size_file else a.sort()
    b_sorted = b.sort(g=genome_size_file) if genome_size_file else b.sort()

    # For center_* summaries, create a windowed-A for mapping
    tmp_center_bed = None
    if summary.startswith("center_"):
        tmp_center_bed = _write_center_window_bed(a_sorted, window_bp=window_bp)
        a_map = BedTool(tmp_center_bed)
        a_map = a_map.sort(g=genome_size_file) if genome_size_file else a_map.sort()
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

def residualize_intensities(
    int_map: dict,
    bed_path: str,
    genome_fa: str,
    *,
    use_length: bool = False,
    resid_open: str = "off",  # {"auto","off","force"}
    output_mode: str = "resid_log",  # {"resid_log","log_corrected","intensity_like"}
):
    """
    Residualize log1p(signal) against GC and optional covariates.

    Model (OLS):
        y_log ~ 1 + gc [+ length] [+ log1p(open)]

    where:
      - y_log = log1p(max(signal, 0))
      - gc is computed from genome_fa over each region
      - length is (end - start)
      - open is taken from the BED peak table itself (if available & selected)

    Parameters
    ----------
    use_length:
        If False, do NOT include length, and resid_open is forced to 'off'.
    resid_open:
        'off'   : never include openness
        'auto'  : include openness only if a usable column exists AND covers all regions
        'force' : require openness; raise if missing / incomplete
    output_mode:
        'resid_log'       : residuals in log space (can be negative)
        'log_corrected'   : residuals re-centered onto an intensity-like log scale (clamped >=0)
        'intensity_like'  : expm1(log_corrected) (nonnegative)

    Returns
    -------
    (out_map, coef_map, formula)

    Prints a coefficient table to stdout.
    """
    import numpy as np

    output_mode = str(output_mode).lower()
    if output_mode not in {"resid_log", "log_corrected", "intensity_like"}:
        raise ValueError("output_mode must be one of {'resid_log','log_corrected','intensity_like'}")

    resid_open = str(resid_open).lower()
    if resid_open not in {"auto", "off", "force"}:
        raise ValueError("resid_open must be one of {'auto','off','force'}")

    if not use_length:
        # Per your convention: openness only makes sense when we're also using length.
        resid_open = "off"

    fetch = _make_fasta_fetcher(genome_fa)
    regions = list(int_map.keys())
    if not regions:
        raise ValueError("No intensities loaded; int_map is empty")

    n = len(regions)

    # covariates + response
    gc = np.zeros(n, dtype=float)
    length = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)

    for i, r in enumerate(regions):
        chrom, start, end = _parse_region_key(r)
        seq = fetch(chrom, start, end)
        gc[i] = _gc_fraction(seq)
        length[i] = float(end - start)
        y[i] = float(int_map[r])

    y_log = np.log1p(np.maximum(y, 0.0))

    # Openness from the BED itself (optional)
    use_open = False
    open_vec = None
    open_col = None

    if resid_open != "off":
        open_map, open_col0, open_missing = _load_openness_from_bed(bed_path)
        if not open_map:
            if resid_open == "force":
                raise ValueError("Openness requested (force) but no numeric openness column was detected in --bed.")
        else:
            tmp = np.array([open_map.get(r, np.nan) for r in regions], dtype=float)
            miss = int(np.isnan(tmp).sum())
            if miss > 0:
                if resid_open == "force":
                    raise ValueError(f"Openness requested (force) but missing for {miss}/{n} regions.")
                # auto: ignore openness
            else:
                use_open = True
                open_col = open_col0
                open_vec = np.log1p(np.maximum(tmp, 0.0))

    # Design matrix
    cols = [np.ones(n, dtype=float), gc]
    names = ["intercept", "gc"]
    formula = "y_log ~ 1 + gc"

    if use_length:
        cols.append(length)
        names.append("length")
        formula += " + length"

    if use_open:
        cols.append(open_vec)
        names.append("log1p(open)")
        formula += " + log1p(open)"

    X = np.vstack(cols).T  # n x p

    # Fit OLS via least squares
    beta, *_ = np.linalg.lstsq(X, y_log, rcond=None)
    yhat = X @ beta
    resid = y_log - yhat

    # Print coefficient table
    print(f"\nCoefficients ({formula}):")
    for nm, b in zip(names, beta):
        print(f"  {nm}: {float(b)}")

    if resid_open == "off":
        print("[Info] openness covariate: OFF")
    else:
        if use_open:
            print(f"[Info] openness covariate: ON (column='{open_col}', coverage=100%)")
        else:
            print("[Info] openness covariate: not used (incomplete coverage or no usable column)")

    # Build residual map
    out_map = {r: float(resid[i]) for i, r in enumerate(regions)}
    coef_map = {names[i]: float(beta[i]) for i in range(len(names))}

    if output_mode in {"log_corrected", "intensity_like"}:
        # Re-center residuals to an "intensity-like" log scale by adding the fitted value at mean covariates.
        # This keeps the covariate-corrected values comparable in scale to log1p(signal).
        cov_means = []
        cov_means.append(1.0)               # intercept
        cov_means.append(float(np.mean(gc)))
        if use_length:
            cov_means.append(float(np.mean(length)))
        if use_open:
            cov_means.append(float(np.mean(open_vec)))

        ref = float(np.dot(beta, np.array(cov_means, dtype=float)))

        # corrected log value per region; clamp >=0 for "intensity-like" behavior
        log_corr = {r: max(0.0, float(v) + ref) for r, v in out_map.items()}

        if output_mode == "log_corrected":
            out_map = log_corr
        else:
            out_map = {r: float(np.expm1(v)) for r, v in log_corr.items()}

    return out_map, coef_map, formula


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
    parser.add_argument("--genome_size_file", default=None, help="Optional genome chrom sizes file for bedtools sort (-g)")
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

    # Residualization options (only used when --residualize is set)
    parser.add_argument(
        "--resid-use-length",
        action="store_true",
        help="Include region length as a covariate during residualization (default: off).",
    )
    parser.add_argument(
        "--resid-open",
        default="off",
        choices=("auto", "off", "force"),
        help="Include openness covariate from --bed during residualization: auto/off/force. "
             "Forced to 'off' unless --resid-use-length is set.",
    )
    parser.add_argument(
        "--resid-output",
        default="intensity_like",
        choices=("resid_log", "log_corrected", "intensity_like"),
        help="Residualization output scale: "
             "resid_log (raw residuals in log space), "
             "log_corrected (recentered log scale), "
             "intensity_like (nonnegative intensity scale).",
    )

    # NEW:
    parser.add_argument(
        "--summary",
        default="max",  
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
            genome_size_file=args.genome_size_file,
            keep_temp=args.keep_temp,
        )
    else:
        run_bedtools_map_pybedtools(
            args.bed,
            args.signal,
            tmp_out,
            summary=args.summary,
            window_bp=args.window_bp,
            genome_size_file=args.genome_size_file,
            keep_temp=args.keep_temp,
        )

    if args.residualize:
        if not args.genome_fasta:
            raise SystemExit("[Error] --genome-fasta is required when --residualize is set.")

        sig_map = _read_two_col_map(tmp_out)

        # Enforce: openness only when length is included
        resid_open = args.resid_open
        if not args.resid_use_length:
            if resid_open != "off":
                print("[Info] --resid-open ignored because --resid-use-length is off (forcing resid_open='off').")
            resid_open = "off"

        resid_map, _, _ = residualize_intensities(
            sig_map,
            bed_path=args.bed,
            genome_fa=args.genome_fasta,
            use_length=args.resid_use_length,
            resid_open=resid_open,
            output_mode=args.resid_output,
        )

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