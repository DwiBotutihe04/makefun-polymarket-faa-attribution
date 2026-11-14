# polymarket-faa-attribution-01 entry (Python 3.13, standard library only).
#
# Model: persistence with an empirically calibrated, asymmetric spread.
#
# For each market, let x be its last price in the case (the price at the
# checkpoint) and d the one-minute price moves over the trailing 24 hours
# of the case's history (all moves when the history is shorter).  For each
# quantile level tau in {0.05, 0.25, 0.5, 0.75, 0.95} let Q_tau be the
# empirical tau-quantile of those moves (linear interpolation).  The
# forecast for minute h = 1..240 after the checkpoint is
#
#   q_tau(h) = clamp(x + sgn(tau) * max(|Q_tau|, |z_tau| * FLOOR)
#                     * h ** ALPHA, 0, 1),  rounded to 6 decimals,
#
# where sgn(tau) is -1 below the median and +1 above it, the median row is
# x itself, z_tau is the standard normal quantile at tau, FLOOR = 0.0005
# price units per sqrt-minute and ALPHA = 0.45.
#
# Why this shape (calibrated on live FAA data fetched 2026-11-15, the
# morning the competition opened; see BUILD-NOTES.md):
# - The four FAA markets crashed from their opening highs in their first
#   hours (Iran 0.515 -> 0.021) and have since gone almost silent: in the
#   15 hours before the open, 96% of minutes did not move at all and the
#   typical non-zero tick is 0.0005.  A fixed house-style sigma of 0.0025
#   is far too wide for that regime; on a rolling backtest over the live
#   FAA history it scores a pinball loss of 0.27, while this model scores
#   ~0.15.
# - Moves are strongly asymmetric: over 4 hours the pooled 5th percentile
#   of the price change is about -0.03 and the 95th about +0.0075 (prices
#   keep drifting down from the crash, rarely up).  Empirical quantiles
#   of the 1-minute moves capture that asymmetry per market; a symmetric
#   normal spread cannot.
# - The FLOOR keeps a minimal spread alive when a market has been dead
#   (all recent moves 0), so a sudden jump does not find every quantile
#   sitting exactly on x.  A jump stress test shows this model loses only
#   ~5% more than the house baseline inside a jump case, while winning
#   ~40% in the calm regime.
# - The jump guard (see JUMP_* below) widens the outer tails when the
#   trailing window contains a large one-minute jump: the empirical 5% /
#   95% quantiles of 1-minute moves are dominated by zeros in the calm
#   regime and do not price the risk that the market reprices in a
#   discrete step again.  The widening is proportional to how far the
#   jump exceeded the threshold, capped, so a single old crash cannot
#   blow the tails out.
# - ALPHA = 0.45 (between the random-walk 0.5 and the fitted ~0.4): the
#   measured 240-minute moves grow more slowly than sqrt(h) would imply,
#   because most of the movement is slow drift plus sparse ticks rather
#   than a random walk.
# - Drift is 0: a trailing-mean drift term was tested and made the
#   backtest strictly worse, so the median stays at the last price.
#
# The program is fully deterministic (no randomness, no network, no clock):
# the same case always produces the same bytes.  It reads every *.in file
# in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
# $MAKEFUN_OUTPUT_DIR (else /output).

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the levels above, to 6 decimals.
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

# Trailing window for the empirical move distribution, in minutes.
WINDOW_MINUTES = 1440  # 24 hours
# Minimum spread, in price units per (minute ** ALPHA), per unit |z|.
FLOOR_SIGMA = 0.0005
# Horizon exponent for the spread: h ** ALPHA.
ALPHA = 0.45
# Fallback spread (house baseline) when a market has too little history
# to estimate its own move distribution.
FALLBACK_SIGMA = 0.0025
MIN_MOVES = 10
# Jump guard: a one-minute move of at least this size in the trailing
# window means the market can reprice in discrete steps.  The outer
# tails are then widened by JUMP_FRAC * (largest |move| - threshold),
# capped at JUMP_CAP (all in price units per (minute ** ALPHA), like the
# other spreads, so the extra grows with h ** ALPHA as well).  The cap
# keeps the cost bounded in the calm regime: widening a tail by e costs
# roughly 100 * 0.05 * e * mean(h ** ALPHA) per pinball item on that
# level, so e = 0.0002 costs ~0.003 of mean loss even if the guard is
# active at every checkpoint.
JUMP_THRESHOLD = 0.05
JUMP_FRAC = 0.002
JUMP_CAP = 0.0002


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation).

    Only used if a case ever asks for quantile levels other than the five
    above; the shipped levels use the exact Z table.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _z_values(quantiles):
    zs = []
    for tau in quantiles:
        for known_tau, known_z in zip(QUANTILES, Z):
            if tau == known_tau:
                zs.append(known_z)
                break
        else:
            zs.append(_norm_ppf(tau))
    return zs


def _empirical_quantile(sorted_moves, p):
    """Linear-interpolation percentile of a sorted list (deterministic)."""
    n = len(sorted_moves)
    if n == 1:
        return sorted_moves[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_moves[lo] * (1.0 - frac) + sorted_moves[hi] * frac


def _clamp6(x):
    """Clamp to [0, 1] and round to 6 decimals (never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def forecast_market(prices, horizon, quantiles, zs):
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    moves = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    # Trailing window: the most recent regime is what the next hours
    # resemble; older moves (e.g. the opening crash) do not describe it.
    if len(moves) > WINDOW_MINUTES:
        moves = moves[-WINDOW_MINUTES:]
    if len(moves) >= MIN_MOVES:
        ordered = sorted(moves)
        spreads = []
        for tau, z in zip(quantiles, zs):
            q = _empirical_quantile(ordered, tau)
            spreads.append(max(abs(q), abs(z) * FLOOR_SIGMA))
        # Jump guard: widen only the outer tails, in proportion to the
        # largest jump in the trailing window above JUMP_THRESHOLD.
        # The empirical quantiles above barely see a rare jump (it sits
        # far out in the tail of a mostly-zero move distribution), so
        # without this the forecast would price a repeat jump at ~zero.
        if len(spreads) >= 2:
            jump = max(abs(m) for m in moves)
            if jump >= JUMP_THRESHOLD:
                extra = min(JUMP_FRAC * (jump - JUMP_THRESHOLD), JUMP_CAP)
                spreads[0] += extra
                spreads[-1] += extra
    else:
        # Too little history to estimate a distribution: fall back to
        # the house baseline's symmetric normal spread.
        spreads = [abs(z) * FALLBACK_SIGMA for z in zs]
    rows = []
    for h in range(1, horizon + 1):
        scale = h ** ALPHA
        row = []
        for j, tau in enumerate(quantiles):
            if tau == 0.5:
                v = x
            elif tau < 0.5:
                v = x - spreads[j] * scale
            else:
                v = x + spreads[j] * scale
            row.append(_clamp6(v))
        # Rounding can only tie neighbours, never invert them, but
        # enforce non-decreasing explicitly so the format always holds.
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in input_dir.iterdir()
                   if f.is_file() and f.name.endswith(".in"))
    for path in files:
        case_id = path.name[:-len(".in")]
        with path.open("r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        quantiles = [float(t) for t in case["quantiles"]]
        zs = _z_values(quantiles)
        forecasts = []
        for market in case["markets"]:
            rows = forecast_market(market["prices"], horizon,
                                   quantiles, zs)
            forecasts.append({"id": market["id"], "q": rows})
        out = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        with (output_dir / (case_id + ".out")).open("w", encoding="utf-8") as fh:
            fh.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
