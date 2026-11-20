"""Validate the VRS-only model against the original HLTV model.

For each finished target stage, the event structure (teams, Round 1
pairings, format, previous-stage carryover, actual results) comes from the
bundled HLTV cache, read offline. Only the team strengths differ:

  hltv  build_team_strengths()      (original model, bundled HLTV cache)
  vrs   build_team_strengths_vrs()  (Valve's public VRS data only)

Reported per stage:
  - Spearman rank correlation of team Elo between the two models
  - mean absolute difference of P(3-0), P(advance), P(0-3) in points
  - how many of the 10 ticket picks agree
  - correct picks of each top ticket against the actual results

Usage:
    python compare_vrs.py --n-sims 5000
"""

from __future__ import annotations

import argparse
import math

from backtest import (
    TARGETS,
    baseline_vrs_ticket,
    detect_all_bo3,
    first_match_cutoff,
    score_ticket_correct,
    swiss_truth,
)
from hltv_api import DiskCache, HLTVClient
from pickem.model import build_team_strengths
from pickem.optimize import best_tickets, marginal_probs
from pickem.run import collect_h2h, extract_r1_pairings, previous_stage_records
from pickem.swiss import simulate_swiss_with_sims
from pickem.vrs_model import build_team_strengths_vrs


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def ticket_slots(ticket) -> set[tuple[str, int]]:
    return (
        {("3-0", t) for t in ticket.three_oh}
        | {("adv", t) for t in ticket.advancers}
        | {("0-3", t) for t in ticket.zero_three}
    )


OUTCOMES = ("3-0", "adv", "out", "0-3")


def outcome_of(rec: tuple[int, int]) -> str:
    w, l = rec
    if (w, l) == (3, 0):
        return "3-0"
    if (w, l) == (0, 3):
        return "0-3"
    return "adv" if w >= 3 else "out"


def log_loss(marg: dict, truth: dict, ids: list[int]) -> float:
    """Mean negative log-likelihood of each team's actual pickem outcome."""
    total = 0.0
    for t in ids:
        m = marg[t]
        probs = {
            "3-0": m["3-0"],
            "adv": m["3-1"] + m["3-2"],
            "0-3": m["0-3"],
            "out": m["2-3"] + m["1-3"],
        }
        total -= math.log(max(1e-3, probs[outcome_of(truth.get(t, (0, 0)))]))
    return total / len(ids)


def run_stage(
    client: HLTVClient, event_id: int, name: str, n_sims: int, replay: str = "since_snapshot"
) -> dict | None:
    print(f"\n=== {name} ({event_id}) ===")
    event = client.get_event(event_id)
    if len(event.team_ids) != 16:
        print(f"  skip: expected 16 teams, got {len(event.team_ids)}")
        return None
    cutoff = first_match_cutoff(event)
    all_bo3 = detect_all_bo3(event)
    carry = previous_stage_records(client, event.name, event.team_ids)
    r1 = extract_r1_pairings(client, event_id)
    r1 = r1 if len(r1) == 8 else None
    truth = swiss_truth(event)
    print(f"  cutoff {cutoff.isoformat()}  {'all BO3' if all_bo3 else 'mixed BO1/BO3'}  carryover={len(carry)}")

    hltv = build_team_strengths(client, event.team_ids, cutoff, stage_entry_records_by_team=carry)
    hltv_h2h = collect_h2h(client, event.team_ids, cutoff)
    names = {tid: s.name for tid, s in hltv.items()}
    vrs, vrs_h2h = build_team_strengths_vrs(
        names, cutoff, stage_entry_records_by_team=carry, replay=replay
    )
    # "hltv_safe": only the cutoff-filtered HLTV signals (map-level Elo
    # replay with round margins, form, H2H), seeded from the same leak-free
    # VRS snapshot as the vrs model. The bundled cache was fetched on
    # 2026-06-10, after every target stage, so world rank, player ratings,
    # map stats and VRS points on the team pages leak future results.
    safe = build_team_strengths(
        client, event.team_ids, cutoff, stage_entry_records_by_team=carry,
        seed_elo_by_team={t: vrs[t].base_elo - vrs[t].stage_entry_boost for t in vrs},
        vrs_points_by_team={t: vrs[t].vrs_points for t in vrs if vrs[t].vrs_points},
        use_player_trends=False, use_roster_rating=False, use_map_pool=False,
    )

    out: dict = {"name": name}
    tickets = {}
    margs = {}
    for label, strengths, h2h in (
        ("hltv", hltv, hltv_h2h), ("safe", safe, hltv_h2h), ("vrs", vrs, vrs_h2h)
    ):
        records, sims = simulate_swiss_with_sims(
            strengths, n_sims=n_sims, h2h=h2h, seed=42,
            r1_pairings=r1, all_matches_bo3=all_bo3,
        )
        top = best_tickets(records, sims=sims, top_k=1, objective="p_at_least_5")[0]
        tickets[label] = top.ticket
        margs[label] = marginal_probs(records)
        out[f"{label}_correct"] = score_ticket_correct(top.ticket, truth)
        out[f"{label}_p5"] = top.p_at_least_5
        out[f"{label}_ll"] = log_loss(margs[label], truth, list(strengths))

    ids = list(hltv)
    out["spearman"] = spearman([hltv[t].elo for t in ids], [vrs[t].elo for t in ids])
    for key in ("3-0", "advance", "0-3"):
        out[f"mad_{key}"] = 100 * sum(
            abs(margs["hltv"][t][key] - margs["vrs"][t][key]) for t in ids
        ) / len(ids)
    out["overlap"] = len(ticket_slots(tickets["hltv"]) & ticket_slots(tickets["vrs"]))
    out["vrs_baseline"] = score_ticket_correct(baseline_vrs_ticket(event, vrs), truth)

    print(f"  {'team':<20} {'hltv elo':>8} {'vrs elo':>8} {'vrs pts':>8}  {'adv hltv':>8} {'adv vrs':>8}  actual")
    for t in sorted(ids, key=lambda t: -hltv[t].elo):
        w, l = truth.get(t, (0, 0))
        vp = f"{vrs[t].vrs_points:.0f}" if vrs[t].vrs_points else "-"
        print(
            f"  {names[t]:<20} {hltv[t].elo:>8.0f} {vrs[t].elo:>8.0f} {vp:>8}  "
            f"{margs['hltv'][t]['advance']*100:>7.0f}% {margs['vrs'][t]['advance']*100:>7.0f}%  {w}-{l}"
        )
    for label in ("hltv", "vrs"):
        tk = tickets[label]
        fmt = lambda xs: ", ".join(names[t] for t in xs)
        print(f"  {label} ticket ({out[label + '_correct']}/10): 3-0 [{fmt(tk.three_oh)}]  adv [{fmt(tk.advancers)}]  0-3 [{fmt(tk.zero_three)}]")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--n-sims", type=int, default=5_000)
    parser.add_argument(
        "--replay", default="since_snapshot", choices=("since_snapshot", "last_n", "none"),
        help="which series update Elo on top of the VRS seed",
    )
    args = parser.parse_args()
    client = HLTVClient(
        cache=DiskCache(base_dir=args.cache_dir, ignore_ttl=True),
        proxy_pool=False,
        offline=True,
    )
    rows = [r for t in TARGETS if (r := run_stage(client, t["id"], t["name"], args.n_sims, args.replay))]

    print("\n=== HLTV model vs VRS-only model ===")
    print(f"  replay mode: {args.replay}")
    print(f"  {'stage':<32} {'rho':>5} {'d3-0':>5} {'dAdv':>5} {'d0-3':>5} {'same':>5} {'hltv':>5} {'vrs':>5} {'base':>5} {'safe':>5} {'LLhltv':>7} {'LLsafe':>7} {'LLvrs':>7}")
    for r in rows:
        print(
            f"  {r['name'][:32]:<32} {r['spearman']:>5.2f} {r['mad_3-0']:>5.1f} {r['mad_advance']:>5.1f} "
            f"{r['mad_0-3']:>5.1f} {r['overlap']:>3}/10 {r['hltv_correct']:>3}/10 {r['vrs_correct']:>3}/10 {r['vrs_baseline']:>3}/10"
            f" {r['safe_correct']:>3}/10 {r['hltv_ll']:>7.3f} {r['safe_ll']:>7.3f} {r['vrs_ll']:>7.3f}"
        )
    n = len(rows)
    print(
        f"  {'AVERAGE':<32} {sum(r['spearman'] for r in rows)/n:>5.2f} "
        f"{sum(r['mad_3-0'] for r in rows)/n:>5.1f} {sum(r['mad_advance'] for r in rows)/n:>5.1f} "
        f"{sum(r['mad_0-3'] for r in rows)/n:>5.1f} {sum(r['overlap'] for r in rows)/n:>5.1f} "
        f"{sum(r['hltv_correct'] for r in rows)/n:>5.1f} {sum(r['vrs_correct'] for r in rows)/n:>5.1f} "
        f"{sum(r['vrs_baseline'] for r in rows)/n:>5.1f} {sum(r['safe_correct'] for r in rows)/n:>5.1f} "
        f"{sum(r['hltv_ll'] for r in rows)/n:>7.3f} {sum(r['safe_ll'] for r in rows)/n:>7.3f} "
        f"{sum(r['vrs_ll'] for r in rows)/n:>7.3f}"
    )
    print("  rho = Spearman of team Elo; d* = mean |difference| in percentage points;")
    print("  same = ticket picks shared; hltv/vrs/base = correct picks vs actual results")
    print("  safe = HLTV map-level replay/form/H2H only, leak-free VRS seed (no rank/players/map stats)")
    print("  LL = mean log-loss of each team's actual outcome (3-0/adv/out/0-3), lower is better;")
    print(f"       uniform guess = {math.log(4):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
