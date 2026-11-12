#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (hourly-ish snapshots of the Up-and-Coming chart:
rank + playing per game) and /input/task.json (the scored 72-hour window), and
writes /output/forecast.json: for each of the top 10 places, a probability for
each named game being at that place, plus a top-10 guess.

Method: a Monte Carlo over continuous ranks. Each candidate game carries a
latent rank that steps hourly with (a) its own damped trajectory drift,
estimated from its observed rank slope, and (b) rank-band diffusion whose
volatility is measured from the history itself. New games spawn as entrants
at the history's measured entry rate and first-appearance rank distribution;
games whose publication date is known (details.created) leave for good at
28 days old. Sorting the latent ranks each simulated hour gives a chart; the
fraction of (simulation, window hour) pairs in which a game holds place k is
its probability for place k -- the honest time-averaged marginal the log
score rewards. Per place, a game is only named when that marginal beats what
it would receive from the place's unnamed share, which is left for the
entrant cohort and the games not named.

Deterministic: one fixed seed, sorted iteration everywhere, stdlib only.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261113
NSIM = 4000
DRIFT_TAU_H = 30.0        # a game's rank slope decays with this time constant
SLOPE_CAP = 2.5           # ranks per hour, before damping
FAT_PROB = 0.08           # fat-tail mixture weight for rank steps
FAT_MULT = 2.2
MIN_ROW_P = 0.0012        # never emit a row under this probability
RETURN_WINDOW_H = 36.0    # games last seen this recently may still be named
RETURN_RANK_PAD = 6.0

# Rank-diffusion sd per sqrt(hour), measured on the practice history and
# re-measured in-program when the history is long (see band_sigma()).
SIGMA_POINTS = [(5.0, 0.75), (15.0, 1.68), (25.0, 2.28), (35.0, 2.83),
                (50.0, 3.33), (75.0, 3.51), (95.0, 3.75)]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 3600.0


def band_sigma(r, points):
    if r <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if r <= x1:
            return y0 + (y1 - y0) * (r - x0) / (x1 - x0)
    return points[-1][1]


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    snaps = []
    seen_at = set()
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not obj.get("ok", True):
                continue
            rows = obj.get("up_and_coming") or []
            if not rows:
                continue
            t = parse_ts(obj["at"])
            if t in seen_at:
                snaps = [s for s in snaps if s[0] != t]
            seen_at.add(t)
            ranks = {}
            for r in rows:
                try:
                    ranks[int(r["universe_id"])] = float(r["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
            created = {}
            det = obj.get("details")
            if isinstance(det, dict):
                for gid_s, d in det.items():
                    if isinstance(d, dict) and d.get("created"):
                        try:
                            created[int(gid_s)] = parse_ts(d["created"])
                        except (ValueError, TypeError):
                            pass
            snaps.append((t, ranks, created))
    snaps.sort(key=lambda s: s[0])
    return task, snaps


def measure_sigma_points(snaps):
    """Re-measure rank-band diffusion from consecutive snapshot pairs."""
    bands = {}
    for (t0, r0, _), (t1, r1, _) in zip(snaps, snaps[1:]):
        dh = t1 - t0
        if dh <= 0:
            continue
        for g, a in r0.items():
            if g in r1:
                bands.setdefault(min(int(a // 10) * 10, 90), []).append(
                    (r1[g] - a) / math.sqrt(dh))
    pts = []
    for lo in sorted(bands):
        v = bands[lo]
        if len(v) >= 40:
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
            pts.append((lo + 5.0, sd))
    if len(pts) >= 4:
        # keep the table monotone-ish and anchored to the measured values
        return pts
    return SIGMA_POINTS


def game_features(snaps, t_last):
    """Per-game slope (ranks/hour, shrunk) and own diffusion sd."""
    obs = {}
    for t, ranks, _ in snaps:
        for g, r in ranks.items():
            obs.setdefault(g, []).append((t, r))
    slopes = {}
    own_sd = {}
    for g, series in obs.items():
        series.sort()
        # weighted least squares slope over the recent past
        sw = sx = sy = sxx = sxy = 0.0
        for t, r in series:
            w = math.exp(-(t_last - t) / 48.0)
            sw += w
            sx += w * t
            sy += w * r
            sxx += w * t * t
            sxy += w * t * r
        den = sw * sxx - sx * sx
        slope = (sw * sxy - sx * sy) / den if den > 1e-9 else 0.0
        # shrink by the effective number of observations
        n = len(series)
        slope *= n / (n + 2.5)
        slopes[g] = max(-SLOPE_CAP, min(SLOPE_CAP, slope))
        ds = []
        for (t0, r0), (t1, r1) in zip(series, series[1:]):
            if t1 > t0:
                ds.append((r1 - r0) / math.sqrt(t1 - t0))
        if len(ds) >= 3:
            m = sum(ds) / len(ds)
            own_sd[g] = math.sqrt(sum((x - m) ** 2 for x in ds) / (len(ds) - 1))
    return obs, slopes, own_sd


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, snaps = load_input(input_dir)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    window = task.get("window", {})
    t_from = parse_ts(window["from"]) if "from" in window else snaps[-1][0]
    t_until = parse_ts(window["until"]) if "until" in window else t_from + 72.0

    t_last, last_ranks, _ = snaps[-1]
    chart_size = len(last_ranks)
    sigma_points = measure_sigma_points(snaps)
    obs, slopes, own_sd = game_features(snaps, t_last)

    # known publication times (age-out at +28 days), from any snapshot's details
    created = {}
    for _, _, c in snaps:
        for g, ct in c.items():
            created.setdefault(g, ct)

    # candidate set: the closing chart, plus games seen in the last
    # RETURN_WINDOW_H hours that have since slipped off it
    cand = {}
    for g, r in last_ranks.items():
        cand[g] = r
    for g, series in obs.items():
        if g not in cand and t_last - series[-1][0] <= RETURN_WINDOW_H:
            cand[g] = series[-1][1] + RETURN_RANK_PAD
    cand_ids = sorted(cand)
    n_cand = len(cand_ids)
    idx = {g: i for i, g in enumerate(cand_ids)}
    r0 = [cand[g] for g in cand_ids]
    slope0 = [slopes.get(g, 0.0) for g in cand_ids]
    vol0 = []
    for g in cand_ids:
        b = band_sigma(cand[g], sigma_points)
        o = own_sd.get(g)
        if o is not None:
            b = math.sqrt((o * o * 3.0 + b * b * 4.0) / 7.0)
        vol0.append(max(0.35, b))
    exit_at = []
    for g in cand_ids:
        ct = created.get(g)
        exit_at.append(ct + 28.0 * 24.0 if ct is not None else None)

    # entrant process, measured from the history itself
    first_rank = []
    seen = set()
    for t, ranks, _ in snaps:
        for g in sorted(ranks, key=lambda g: ranks[g]):
            if g not in seen:
                seen.add(g)
                first_rank.append(ranks[g])
    span = max(1.0, snaps[-1][0] - snaps[0][0])
    n_extra = len(seen) - len(snaps[0][1])
    lam = max(0.05, n_extra / span)  # new chart games per hour
    first_rank.sort()
    # entrants are growth-selected: draw their drifts from the slopes games
    # actually showed in their first day on the chart, not from all incumbents
    early_slopes = []
    for g, series in obs.items():
        t0, r_first = series[0]
        r_late = None
        for t, r in series:
            if t - t0 <= 24.0:
                r_late = (t, r)
        if r_late is not None and r_late[0] - t0 >= 6.0:
            early_slopes.append((r_late[1] - r_first) / (r_late[0] - t0))
    slope_pool = sorted(early_slopes) if len(early_slopes) >= 10 else \
        sorted(slopes.values()) or [0.0]

    rng = random.Random(SEED)
    gauss = rng.gauss
    rnd = rng.random
    # tally[game index][place] ; entrant_tally[place]
    tally = [[0] * places for _ in range(n_cand)]
    ent_tally = [0] * places
    n_slots = 0

    n_window_slots = int(round(t_until - t_from))
    slot_targets = [t_from + m for m in range(n_window_slots)]
    total_steps = int(math.ceil(t_until - t_last))
    for _sim in range(NSIM):
        r = list(r0)
        alive = [True] * n_cand
        ent_r = []      # entrant latent ranks
        ent_slope = []  # entrant drifts
        ent_vol = []    # entrant vols
        next_slot = 0
        for step in range(1, total_steps + 1):
            t_now = t_last + step
            damp = math.exp(-(step - 1) / DRIFT_TAU_H)
            # incumbents
            for i in range(n_cand):
                if not alive[i]:
                    continue
                ea = exit_at[i]
                if ea is not None and t_now >= ea:
                    alive[i] = False
                    r[i] = 1e9
                    continue
                z = gauss(0.0, 1.0)
                if rnd() < FAT_PROB:
                    z *= FAT_MULT
                ri = r[i] + slope0[i] * damp + vol0[i] * z
                r[i] = 0.35 if ri < 0.35 else ri
            # entrants: Poisson(lam) arrivals
            x = rnd()
            # Poisson CDF for lam (small): walk the probabilities
            p0 = math.exp(-lam)
            cum = p0
            k_new = 0
            pk = p0
            while x > cum and k_new < 6:
                k_new += 1
                pk *= lam / k_new
                cum += pk
            for _e in range(k_new):
                u = rnd()
                j = int(u * len(first_rank))
                if j >= len(first_rank):
                    j = len(first_rank) - 1
                ent_r.append(first_rank[j] + rnd())
                sj = int(rnd() * len(slope_pool))
                if sj >= len(slope_pool):
                    sj = len(slope_pool) - 1
                ent_slope.append(slope_pool[sj])
                ent_vol.append(band_sigma(ent_r[-1], sigma_points))
            # move entrants
            keep = []
            for j in range(len(ent_r)):
                z = gauss(0.0, 1.0)
                if rnd() < FAT_PROB:
                    z *= FAT_MULT
                rj = ent_r[j] + ent_slope[j] * damp + ent_vol[j] * z
                if rj < 0.35:
                    rj = 0.35
                if rj <= 150.0:
                    keep.append(j)
                    ent_r[j] = rj
            if len(keep) < len(ent_r):
                ent_r = [ent_r[j] for j in keep]
                ent_slope = [ent_slope[j] for j in keep]
                ent_vol = [ent_vol[j] for j in keep]
            # tally each scored slot once, at the first simulated state whose
            # time reaches the slot's start
            while next_slot < n_window_slots and t_now >= slot_targets[next_slot] - 1e-9:
                next_slot += 1
                n_slots += 1
                order = [(r[i], i) for i in range(n_cand) if alive[i]]
                order += [(ent_r[j], -1 - j) for j in range(len(ent_r))]
                order.sort()
                for place in range(min(places, len(order))):
                    who = order[place][1]
                    if who >= 0:
                        tally[who][place] += 1
                    else:
                        ent_tally[place] += 1
    # n_slots counts (simulation, slot) pairs, so it is the tally denominator
    denom = float(max(1, n_slots))
    marg = [[tally[i][k] / denom for k in range(places)] for i in range(n_cand)]
    ent_share = [ent_tally[k] / denom for k in range(places)]

    probabilities = []
    for k in range(places):
        order_i = sorted(range(n_cand), key=lambda i: (-marg[i][k], cand_ids[i]))
        # name from the top while the marginal beats the unnamed share the
        # game would otherwise get
        named = []
        excluded_mass = ent_share[k]
        for i in order_i:
            excluded_mass += marg[i][k]
        included_mass = 0.0
        for i in order_i:
            rest = excluded_mass - marg[i][k]
            n_unnamed = chart_size - len(named)
            share = rest / n_unnamed if n_unnamed > 0 else 0.0
            if marg[i][k] > share and marg[i][k] >= MIN_ROW_P:
                named.append(i)
                included_mass += marg[i][k]
                excluded_mass = rest
            else:
                break
        for i in named:
            p = marg[i][k]
            p = int(p * 1_000_000) / 1_000_000  # floor to 6 dp: sums stay <= 1
            if p > 0:
                probabilities.append(
                    {"universe_id": cand_ids[i], "place": k + 1, "p": p})

    # top 10 guess: by P(holds a top-10 place), ordered by mean place
    p_top = []
    for i in range(n_cand):
        tot = sum(marg[i])
        if tot > 0:
            mean_place = sum((k + 1) * marg[i][k] for k in range(places)) / tot
            p_top.append((-tot, mean_place, cand_ids[i]))
    p_top.sort()
    top10 = [g for _, _, g in p_top[:places]]
    if len(top10) < places:
        for g in sorted(last_ranks, key=lambda g: last_ranks[g]):
            if g not in top10:
                top10.append(g)
            if len(top10) == places:
                break

    os.makedirs(output_dir, exist_ok=True)
    out = {"probabilities": probabilities, "top10": top10}
    with open(os.path.join(output_dir, out_name), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
