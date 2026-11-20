"""
Team strengths built only from Valve's public VRS data.

This is the HLTV-free counterpart of :func:`pickem.model.build_team_strengths`.
It produces the same :class:`TeamStrength` objects, so the Swiss simulator
and ticket optimizer are reused unchanged.

Signals (mirroring the original model where the data allows):

  - Seed Elo = VRS points from the last snapshot on or before the cutoff,
    rescaled to map-level Elo (see :func:`fit_points_scale`).
  - Series-level Elo replay of the series played between the VRS snapshot
    and the cutoff (older ones are already in the points). The original
    replays maps with round margins; VRS data has only series W/L.
  - Exponentially weighted recent form over the same series.
  - Previous-stage carryover boost and form floor, identical to the
    original.
  - H2H between event teams over the last 6 months.
  - VRS points as the matchup blend prior (``vrs_points``).

Not available from VRS data: roster Rating 3.0 and player trends, map
pools (veto falls back to random maps), round margins, big-event
pedigree.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from .model import TeamStrength
from .vrs_data import norm_name, series_before, snapshot_at_or_before

# Display name in other sources -> name used in Valve's standings.
NAME_ALIASES: dict[str, str] = {
    "betboom": "BETBOOM",
    "navi": "Natus Vincere",
    "nip": "Ninjas in Pyjamas",
    "teamliquid": "Liquid",
    "fazeclan": "FaZe",
    # BC.Game compete as GOSH at the PGL Singapore Major 2026 (gambling
    # sponsor naming rules); same s1mple/asap/Senzu/Magisk/mzinho roster.
    "gosh": "BC.Game",
}

STAGE_BOOST = {(3, 0): 70.0, (3, 1): 38.0, (3, 2): 18.0}
STAGE_FORM_FLOOR = {(3, 0): 0.85, (3, 1): 0.72, (3, 2): 0.62}


def resolve_name(name: str, known: set[str]) -> str | None:
    """Map an external team name to a normalised VRS name, or None."""
    n = norm_name(name)
    if n in known:
        return n
    alias = NAME_ALIASES.get(n)
    if alias and norm_name(alias) in known:
        return norm_name(alias)
    return None


def build_team_strengths_vrs(
    names_by_id: dict[int, str],
    cutoff: datetime,
    *,
    stage_entry_records_by_team: dict[int, tuple[int, int]] | None = None,
    n_recent: int = 10,
    k_series: float = 40.0,
    half_life_days: float = 30.0,
    unranked_points: float = 1000.0,
    points_scale: float | None = None,
    replay: str = "since_snapshot",
    verbose: bool = True,
) -> tuple[dict[int, TeamStrength], dict[tuple[int, int], tuple[int, int]]]:
    """
    Build strengths and the H2H table for the event teams.

    Parameters
    ----------
    names_by_id : {team_id: display name}
        Event participants. Ids are opaque (HLTV ids for backtests, any
        stable ints for new events).
    cutoff : datetime
        Pickem deadline. Only series dated before ``cutoff.date()`` count.
    points_scale : float, optional
        Elo = 1500 + points_scale * (VRS points - 1500). VRS point gaps are
        much wider than map-level Elo gaps, so this is < 1. When omitted it
        is fitted with :func:`fit_points_scale` on data before the cutoff.
    replay : "since_snapshot" | "last_n" | "none"
        Which series get an Elo update on top of the VRS seed: only those
        newer than the snapshot (default; older ones are already in the
        points), each team's last ``n_recent`` series (like the original
        HLTV model), or none.

    Returns
    -------
    (strengths, h2h)
    """
    stage_entry_records_by_team = stage_entry_records_by_team or {}
    cut_day = cutoff.date()
    snap = snapshot_at_or_before(cut_day)
    points = snap.points_by_name()
    all_series = series_before(cut_day)

    known = set(points)
    for _, _, w, l in all_series:
        known.add(norm_name(w))
        known.add(norm_name(l))

    key_by_id: dict[int, str] = {}
    for tid, name in names_by_id.items():
        key = resolve_name(name, known)
        if key is None:
            if verbose:
                print(f"  [vrs] no VRS data for '{name}', treating as unranked")
            key = norm_name(name)
        key_by_id[tid] = key
    id_by_key = {k: tid for tid, k in key_by_id.items()}

    if points_scale is None:
        points_scale, n_fit = fit_points_scale(cut_day)
        if verbose:
            print(f"  [vrs] points->Elo scale {points_scale:.2f} (fitted on {n_fit} pre-cutoff series)")

    def pts(key: str) -> float:
        return 1500.0 + points_scale * (points.get(key, unranked_points) - 1500.0)

    # Per-team series lists (most recent first)
    per_team: dict[str, list[tuple]] = {k: [] for k in key_by_id.values()}
    h2h_raw: dict[tuple[int, int], list[int]] = {}
    for mid, day, winner, loser in sorted(all_series, key=lambda s: (s[1], s[0]), reverse=True):
        w, l = norm_name(winner), norm_name(loser)
        if w in per_team:
            per_team[w].append((day, l, True))
        if l in per_team:
            per_team[l].append((day, w, False))
        if w in id_by_key and l in id_by_key:
            a, b = id_by_key[w], id_by_key[l]
            lo, hi = (a, b) if a < b else (b, a)
            entry = h2h_raw.setdefault((lo, hi), [0, 0])
            entry[0 if a == lo else 1] += 1

    out: dict[int, TeamStrength] = {}
    cutoff_ord = cut_day.toordinal()
    decay = math.log(2) / half_life_days
    for tid, key in key_by_id.items():
        base = pts(key)
        record = stage_entry_records_by_team.get(tid)
        boost = STAGE_BOOST.get(record, 0.0)
        base += boost

        history = per_team.get(key, [])
        recent = history[:n_recent]
        elo = base
        # Elo replay only for series newer than the VRS snapshot: older
        # results are already inside the VRS points. Oldest -> newest.
        if replay == "since_snapshot":
            newer = [s for s in history if s[0] >= snap.day]
        elif replay == "last_n":
            newer = recent
        else:
            newer = []
        for day, opp, won in reversed(newer):
            expected = 1.0 / (1.0 + 10.0 ** ((pts(opp) - elo) / 400.0))
            elo += k_series * ((1.0 if won else 0.0) - expected)
        wnum = wden = 0.0
        for day, opp, won in recent:
            wt = math.exp(-decay * (cutoff_ord - day.toordinal()))
            wden += wt
            wnum += wt if won else 0.0
        form = wnum / wden if wden else 0.5
        if record in STAGE_FORM_FLOOR:
            form = max(form, STAGE_FORM_FLOOR[record])

        row = snap.row_by_name().get(key)
        out[tid] = TeamStrength(
            team_id=tid,
            name=names_by_id[tid],
            world_rank=row.rank if row else None,
            vrs_rank=row.rank if row else None,
            elo=elo,
            base_elo=base,
            elo_delta=elo - base,
            roster_rating=None,
            recent_form=form,
            sample_n=len(recent),
            map_sample_n=0,
            vrs_points=points.get(key),
            stage_entry_record=record,
            stage_entry_boost=boost,
        )
    h2h = {k: (v[0], v[1]) for k, v in h2h_raw.items()}
    return out, h2h


def fit_points_scale(
    before,
    since=date(2025, 1, 1),
    grid: tuple[float, ...] = tuple(x / 20 for x in range(4, 41)),
) -> tuple[float, int]:
    """
    Fit the factor ``s`` that turns VRS point gaps into map-level Elo gaps.

    For every pair of consecutive snapshots (S_i, S_i+1) published before
    ``before``, take the series played between them by two teams ranked in
    S_i, and score

        p_map    = 1 / (1 + 10 ** (-s * (pts_a - pts_b) / 400))
        p_series = p_map**2 * (3 - 2 * p_map)        (BO3)

    by log-likelihood of the actual winner. Only data published before
    ``before`` is used, so backtests stay leak-free.

    Returns (best_s, n_series).
    """
    from .vrs_data import list_snapshots, load_snapshot

    days = [d for d in list_snapshots() if d < before and (since is None or d >= since)]
    samples: list[float] = []  # point gap, winner minus loser
    for d0, d1 in zip(days, days[1:]):
        p0 = load_snapshot(d0).points_by_name()
        for _, day, w, l in load_snapshot(d1).series:
            if not (d0 <= day < d1):
                continue
            pw, pl = p0.get(norm_name(w)), p0.get(norm_name(l))
            if pw is None or pl is None:
                continue
            samples.append(pw - pl)
    best_s, best_ll = grid[0], -1e18
    for s in grid:
        ll = 0.0
        for gap in samples:
            p = 1.0 / (1.0 + 10.0 ** (-s * gap / 400.0))
            ps = p * p * (3 - 2 * p)
            ll += math.log(max(1e-9, ps))
        if ll > best_ll:
            best_s, best_ll = s, ll
    return best_s, len(samples)
