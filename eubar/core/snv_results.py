"""Within-SNV correction and shared-position result selection."""
import math


def _extract_stat(r: dict) -> float:
    """Try to extract a test statistic from a row dict (z/t)."""
    for k in ("stat", "z", "zval", "zvalue", "t", "tval", "tvalue"):
        v = r.get(k)
        if v is None:
            continue
        try:
            x = float(v)
            if math.isfinite(x):
                return x
        except Exception:
            continue
    return math.nan


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def _parse_ref_alt_from_snv(snv_str: str):
    try:
        allele_part = snv_str.split(":", 2)[2]
        ref, alt = allele_part.split(">", 1)
        ref = ref.strip().upper()
        alt = alt.strip().upper()
        if ref in {"A", "C", "G", "T"} and alt in {"A", "C", "G", "T"}:
            return ref, alt
    except Exception:
        pass
    return None, None


def _holm_adjust(pvals):
    """Return Holm-adjusted p-values, preserving NaNs.

    Holm controls the family-wise error rate under arbitrary dependence.
    This implementation is dependency-free and equivalent to the standard
    step-down Holm procedure.
    """
    out = [math.nan] * len(pvals)
    valid = []
    for i, p in enumerate(pvals):
        p = _safe_float(p)
        if math.isfinite(p):
            valid.append((i, min(max(p, 0.0), 1.0)))

    m = len(valid)
    if m == 0:
        return out

    valid.sort(key=lambda x: x[1])
    running = 0.0
    for rank, (idx, p) in enumerate(valid, start=1):
        adjusted = (m - rank + 1) * p
        running = max(running, adjusted)
        out[idx] = min(running, 1.0)
    return out


def _apply_holm_within_snv(rows, snv_str: str):
    """Annotate rows with ``pval_holm`` for the tests used by --best-pval.

    Families are corrected separately:
      * AFF: ALT-allele AFF tests across motif positions (normally k tests).
      * RAND: REF + ALT RAND tests across motif positions (normally 2*k tests).

    Other rows are left with pval_holm=NaN because they are not part of the
    --best-pval decision for the requested SNV. Raw ``pval`` is never changed.
    """
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    out = [dict(r) for r in rows]
    for r in out:
        r["pval_holm"] = math.nan

    if alt is None:
        return out

    aff_idx = [
        i for i, r in enumerate(out)
        if r.get("label", "AFF") == "AFF" and r.get("allele") == alt
        and math.isfinite(_safe_float(r.get("pval")))
    ]
    aff_adj = _holm_adjust([out[i].get("pval") for i in aff_idx])
    for i, q in zip(aff_idx, aff_adj):
        out[i]["pval_holm"] = q

    rand_alleles = {a for a in (ref, alt) if a is not None}
    rand_idx = [
        i for i, r in enumerate(out)
        if r.get("label", "AFF") == "RAND" and r.get("allele") in rand_alleles
        and math.isfinite(_safe_float(r.get("pval")))
    ]
    rand_adj = _holm_adjust([out[i].get("pval") for i in rand_idx])
    for i, q in zip(rand_idx, rand_adj):
        out[i]["pval_holm"] = q

    return out


def _best_supported_motif_pos(rows, snv_str: str, *, use_holm: bool = False):
    """Choose one motif position per SNP using ALT AFF first, then RAND support.

    AFF candidates are ranked by their raw p-value (Holm preserves this order).
    With ``use_holm=True``, RAND support requires the Holm-adjusted RAND p-value
    to be < 0.05; otherwise the legacy raw RAND p-value is used.
    """
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    if alt is None:
        return None

    by_key = {}
    candidates = []

    for r in rows:
        label = r.get("label", "AFF")
        allele = r.get("allele")
        if label not in ("AFF", "RAND") or allele not in ("A", "C", "G", "T"):
            continue

        motif_pos = r.get("motif_pos")
        try:
            motif_pos = int(motif_pos)
        except Exception:
            continue

        p_raw = _safe_float(r.get("pval"))
        p_holm = _safe_float(r.get("pval_holm"))
        c = _safe_float(r.get("coef"))

        by_key[(motif_pos, label, allele)] = {
            "effect": c,
            "pval": p_raw,
            "pval_holm": p_holm,
        }

        if label == "AFF" and allele == alt and not math.isnan(p_raw):
            candidates.append((p_raw, -abs(c) if not math.isnan(c) else 0.0, motif_pos))

    if not candidates:
        return None

    candidates.sort()

    def _rand_passes(r):
        if r is None:
            return False
        p = r["pval_holm"] if use_holm else r["pval"]
        c = r["effect"]
        return not math.isnan(p) and not math.isnan(c) and p < 0.05 and c > 0.0

    def _passes_rand_support(motif_pos):
        if ref is None:
            return False
        return (
            _rand_passes(by_key.get((motif_pos, "RAND", alt)))
            or _rand_passes(by_key.get((motif_pos, "RAND", ref)))
        )

    for _, _, motif_pos in candidates:
        if _passes_rand_support(motif_pos):
            return motif_pos

    return candidates[0][2]


def _best_pval_summary_rows(rows, snv_str: str, *, use_holm: bool = False):
    """Return three rows for the chosen motif position: AFF alt, RAND ref, RAND alt."""
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    chosen_pos = _best_supported_motif_pos(rows, snv_str, use_holm=use_holm)

    def _find(label, allele):
        if chosen_pos is None:
            return None
        for r in rows:
            if (
                r.get("label", "AFF") == label
                and r.get("allele") == allele
                and _safe_float(r.get("motif_pos")) == float(chosen_pos)
            ):
                raw_p = _safe_float(r.get("pval"))
                holm_p = _safe_float(r.get("pval_holm"))
                return {
                    "type":      label,
                    "allele":    allele,
                    "effect":    _safe_float(r.get("coef")),
                    "pval":      holm_p if use_holm else raw_p,
                    "raw_pval":  raw_p,
                    "holm_pval": holm_p,
                    "stat":      _extract_stat(r),
                    "motif_pos": chosen_pos,
                }
        return None

    def _empty(label, allele):
        return {
            "type":      label,
            "allele":    allele,
            "effect":    math.nan,
            "pval":      math.nan,
            "raw_pval":  math.nan,
            "holm_pval": math.nan,
            "stat":      math.nan,
            "motif_pos": chosen_pos,
        }

    return [
        _find("AFF",  alt) or _empty("AFF",  alt),
        _find("RAND", ref) or _empty("RAND", ref),
        _find("RAND", alt) or _empty("RAND", alt),
    ]
