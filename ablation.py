"""Ablation: how much do the HLTV-only signals matter?

Runs the backtest targets offline from the bundled cache with three
model variants:

  full            the original model
  no_players      no roster Rating 3.0, no player trend / 3-month rating
  vrs_seed        no_players, and Elo seeded from VRS points instead of
                  the HLTV world rank (roughly what Liquipedia + Valve VRS
                  data can provide)

For each variant it prints correct picks per event and how many of the
10 picks agree with the full model's ticket.
"""

from __future__ import annotations

import argparse

from backtest import TARGETS, run_one
from hltv_api import DiskCache, HLTVClient

VARIANTS = {
    "full": ({}, "hltv"),
    "no_players": ({"use_player_trends": False, "use_roster_rating": False}, "hltv"),
    "vrs_seed": ({"use_player_trends": False, "use_roster_rating": False}, "vrs"),
}


def ticket_slots(ticket) -> set[tuple[str, int]]:
    return (
        {("3-0", t) for t in ticket.three_oh}
        | {("adv", t) for t in ticket.advancers}
        | {("0-3", t) for t in ticket.zero_three}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--n-sims", type=int, default=5_000)
    args = parser.parse_args()

    client = HLTVClient(
        cache=DiskCache(base_dir=args.cache_dir, ignore_ttl=True),
        proxy_pool=False,
        offline=True,
    )
    results: dict[str, dict[str, dict]] = {}
    for vname, (kwargs, rank_source) in VARIANTS.items():
        print(f"\n######## variant: {vname}")
        results[vname] = {}
        for target in TARGETS:
            r = run_one(
                client, target["id"], target["name"], args.n_sims,
                strength_kwargs=kwargs, rank_source=rank_source,
            )
            if r:
                results[vname][target["name"]] = r

    print("\n=== Ablation summary (correct / 10, overlap with full ticket) ===")
    names = [t["name"] for t in TARGETS]
    print(f"  {'variant':<15}" + "".join(f"{n[:26]:>28}" for n in names) + f"{'avg':>7}")
    for vname, per_event in results.items():
        cells, tot = [], 0
        for n in names:
            r = per_event.get(n)
            full = results["full"].get(n)
            if not r or not full:
                cells.append(f"{'-':>28}")
                continue
            overlap = len(ticket_slots(r["ticket"]) & ticket_slots(full["ticket"]))
            tot += r["model_correct"]
            cells.append(f"{r['model_correct']:>2}/10 (same {overlap:>2}/10)".rjust(28))
        print(f"  {vname:<15}" + "".join(cells) + f"{tot/len(names):>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
