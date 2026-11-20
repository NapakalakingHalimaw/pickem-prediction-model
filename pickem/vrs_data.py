"""
Loader for Valve's public Regional Standings (VRS) data.

Valve publishes its standings at
https://github.com/ValveSoftware/counter-strike_regional_standings
roughly once a month. Every snapshot has:

  - ``live/<year>/standings_global_<date>.md``: rank, points, team, roster
  - ``live/<year>/details/<date>/<rank>--<team>--<roster>.md``: every
    series the roster played in the 6 months before the snapshot
    (date, opponent, W/L), plus the VRS factor breakdown.

That gives us, without touching HLTV, a dated series-level match history
for every ranked team and the VRS points at each snapshot. It does not
include map names or round scores.

Call :func:`sync_repo` once to clone/update the repository into
``data/vrs_repo`` (plain ``git``; the repo is ~25 MB).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

REPO_URL = "https://github.com/ValveSoftware/counter-strike_regional_standings.git"
DEFAULT_REPO_DIR = Path(__file__).resolve().parent.parent / "data" / "vrs_repo"


def sync_repo(repo_dir: Path = DEFAULT_REPO_DIR) -> None:
    """Clone the VRS repository, or fast-forward it if already present."""
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "-C", str(repo_dir), "pull", "-q", "--ff-only"], check=True)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", REPO_URL, str(repo_dir)], check=True
        )


def norm_name(name: str) -> str:
    """Normalise a team name for matching across sources."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class StandingRow:
    rank: int
    points: float
    name: str
    roster: list[str]
    details_path: Path | None


@dataclass
class Series:
    """One series from one team's point of view."""

    match_id: int
    day: date
    team: str
    opponent: str
    won: bool


@dataclass
class Snapshot:
    day: date
    standings: list[StandingRow]
    # Unique series (deduplicated by Valve's per-snapshot match id), each
    # stored once as (match_id, day, winner, loser).
    series: list[tuple[int, date, str, str]] = field(default_factory=list)

    def points_by_name(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in self.standings:
            out.setdefault(norm_name(row.name), row.points)
        return out

    def row_by_name(self) -> dict[str, StandingRow]:
        out: dict[str, StandingRow] = {}
        for row in self.standings:
            out.setdefault(norm_name(row.name), row)
        return out


_SNAPSHOT_RE = re.compile(r"standings_global_(\d{4})_(\d{2})_(\d{2})\.md$")


def list_snapshots(repo_dir: Path = DEFAULT_REPO_DIR) -> list[date]:
    days = []
    for f in (repo_dir / "live").glob("*/standings_global_*.md"):
        m = _SNAPSHOT_RE.search(f.name)
        if m:
            days.append(date(int(m[1]), int(m[2]), int(m[3])))
    return sorted(days)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


@lru_cache(maxsize=None)
def load_snapshot(day: date, repo_dir: Path = DEFAULT_REPO_DIR) -> Snapshot:
    tag = day.strftime("%Y_%m_%d")
    year_dir = repo_dir / "live" / str(day.year)
    standings_file = year_dir / f"standings_global_{tag}.md"
    rows: list[StandingRow] = []
    for line in standings_file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        c = _cells(line)
        if len(c) < 4 or not c[0].isdigit():
            continue
        link = re.search(r"\((details/[^)]+)\)", line)
        rows.append(
            StandingRow(
                rank=int(c[0]),
                points=float(c[1]),
                name=c[2],
                roster=[p.strip() for p in c[3].split(",") if p.strip()],
                details_path=(year_dir / link[1]) if link else None,
            )
        )

    seen: dict[int, tuple[int, date, str, str]] = {}
    for row in rows:
        if row.details_path is None or not row.details_path.exists():
            continue
        for s in _parse_details(row.details_path, row.name):
            if s.match_id in seen:
                continue
            winner, loser = (s.team, s.opponent) if s.won else (s.opponent, s.team)
            seen[s.match_id] = (s.match_id, s.day, winner, loser)
    series = sorted(seen.values(), key=lambda x: (x[1], -x[0]))
    return Snapshot(day=day, standings=rows, series=series)


def _parse_details(path: Path, team_name: str) -> list[Series]:
    """Parse the per-match factor table of a team details file."""
    out: list[Series] = []
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Match Played"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if out:
                break
            continue
        c = _cells(line)
        if len(c) < 5 or not c[1].isdigit():
            continue
        try:
            day = datetime.strptime(c[2], "%Y-%m-%d").date()
        except ValueError:
            continue
        if c[4] not in ("W", "L"):
            continue
        out.append(
            Series(match_id=int(c[1]), day=day, team=team_name, opponent=c[3], won=c[4] == "W")
        )
    return out


def snapshot_at_or_before(cutoff: date, repo_dir: Path = DEFAULT_REPO_DIR) -> Snapshot:
    """Latest snapshot published on or before ``cutoff`` (for VRS points)."""
    days = [d for d in list_snapshots(repo_dir) if d <= cutoff]
    if not days:
        raise ValueError(f"no VRS snapshot on or before {cutoff}")
    return load_snapshot(days[-1], repo_dir)


def series_before(cutoff: date, repo_dir: Path = DEFAULT_REPO_DIR) -> list[tuple[int, date, str, str]]:
    """
    Every series played strictly before ``cutoff``.

    Uses the first snapshot published *after* the cutoff when one exists,
    so matches played between the previous snapshot and the cutoff are
    included. The list is filtered by match date, so nothing played on or
    after the cutoff leaks in. Falls back to the latest snapshot.
    """
    days = list_snapshots(repo_dir)
    after = [d for d in days if d > cutoff]
    snap = load_snapshot(after[0] if after else days[-1], repo_dir)
    return [s for s in snap.series if s[1] < cutoff]
