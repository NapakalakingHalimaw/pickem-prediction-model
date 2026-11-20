"""Predict a Major Swiss stage from Valve's public VRS data (no HLTV).

The event is described by a small JSON file (see data/events/), with the
16 teams, Round 1 pairings, cutoff, format and, for Stages 2/3, the
previous-stage records of the teams that qualified.

Usage:
    python predict_vrs.py data/events/pgl_singapore_2026_stage1.json --n-sims 20000
    python predict_vrs.py <event.json> --sync    # git pull the VRS data first
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from pickem.optimize import best_tickets, explain_ticket, marginal_probs
from pickem.swiss import simulate_swiss_with_sims
from pickem.vrs_data import sync_repo
from pickem.vrs_model import build_team_strengths_vrs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file")
    parser.add_argument("--n-sims", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sync", action="store_true", help="update data/vrs_repo first")
    args = parser.parse_args()

    if args.sync:
        sync_repo()

    with open(args.event_file, encoding="utf-8") as fh:
        ev = json.load(fh)
    cutoff = datetime.fromisoformat(ev["cutoff"].replace("Z", "+00:00"))
    names = {int(t["id"]): t["name"] for t in ev["teams"]}
    carry = {int(k): tuple(v) for k, v in (ev.get("stage_entry_records") or {}).items()}
    r1 = [tuple(p) for p in ev.get("r1_pairings") or []] or None
    all_bo3 = bool(ev.get("all_matches_bo3"))

    print(f"{ev['name']}")
    print(f"  cutoff {cutoff.isoformat()}  format {'all BO3' if all_bo3 else 'mixed BO1/BO3'}")
    strengths, h2h = build_team_strengths_vrs(names, cutoff, stage_entry_records_by_team=carry)

    # Initial seeds for Buchholz ties: from the event file when given
    # ({team_id: seed}), otherwise VRS order.
    if ev.get("initial_seeds"):
        seeds = {int(k): int(v) for k, v in ev["initial_seeds"].items()}
    else:
        order = sorted(strengths, key=lambda t: -(strengths[t].vrs_points or 0))
        seeds = {tid: i + 1 for i, tid in enumerate(order)}

    print(f"\n  {'team':<20} {'VRS #':>6} {'points':>7} {'elo':>6} {'form':>5} {'n':>3}  carry")
    for tid in sorted(strengths, key=lambda t: -strengths[t].elo):
        s = strengths[tid]
        carry_s = f"{s.stage_entry_record[0]}-{s.stage_entry_record[1]}" if s.stage_entry_record else "-"
        pts = f"{s.vrs_points:.0f}" if s.vrs_points else "-"
        print(f"  {s.name:<20} {s.vrs_rank or '-':>6} {pts:>7} {s.elo:>6.0f} {s.recent_form:>5.2f} {s.sample_n:>3}  {carry_s}")
    if r1:
        print("\n  Round 1:")
        for a, b in r1:
            print(f"    {names[a]:<20} vs  {names[b]}")
    if h2h:
        print(f"\n  H2H pairs with history (last 6 months): {len(h2h)}")

    print(f"\nSimulating Swiss x {args.n_sims:,}...")
    records, sims = simulate_swiss_with_sims(
        strengths, n_sims=args.n_sims, h2h=h2h, seed=42,
        r1_pairings=r1, initial_seeds=seeds, all_matches_bo3=all_bo3,
    )
    marg = marginal_probs(records)
    print(f"\n=== Marginal probabilities ===")
    print(f"  {'team':<20}  {'3-0':>5}  {'3-1':>5}  {'3-2':>5}  {'adv':>5}  {'2-3':>5}  {'1-3':>5}  {'0-3':>5}")
    for tid in sorted(names, key=lambda t: -marg[t]["advance"]):
        m = marg[tid]
        print(
            f"  {m['name']:<20}  {m['3-0']*100:>4.1f}%  {m['3-1']*100:>4.1f}%  {m['3-2']*100:>4.1f}%  "
            f"{m['advance']*100:>4.1f}%  {m['2-3']*100:>4.1f}%  {m['1-3']*100:>4.1f}%  {m['0-3']*100:>4.1f}%"
        )

    print(f"\n=== Top tickets by P(>=5 correct) ===")
    for i, score in enumerate(best_tickets(records, sims=sims, top_k=args.top_k, objective="p_at_least_5")):
        print(f"\n#{i+1}  P(>=5 correct) = {score.p_at_least_5*100:.1f}%")
        print(explain_ticket(score.ticket, records, sims=sims, score=score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
