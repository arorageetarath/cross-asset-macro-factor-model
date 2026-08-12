"""
Daily orchestration: collect fresh data, rebuild features, explain today's move,
print the strategist note. Schedule with cron / Task Scheduler / GitHub Actions:

    0 23 * * 1-5  cd /path/to/what-moves-markets && python scripts/daily_update.py

Run from repo root:  python scripts/daily_update.py [--skip-collect]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db  # noqa: E402
from src.attribution import explain_day  # noqa: E402
from src.commentary import contributions_table, write_note  # noqa: E402
from src.features import build_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true", help="use existing warehouse data")
    ap.add_argument("--target", default="RET_SPX")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest)")
    args = ap.parse_args()

    if not args.skip_collect:
        from src.collect import macro, markets
        markets.main()
        try:
            macro.main()
        except SystemExit as e:
            print(f"[warn] {e} — continuing with market data only.")

    conn = db.get_conn()
    features = build_features(conn)
    exp = explain_day(features, target=args.target, date=args.date)

    print("\n" + "=" * 60)
    print("WHAT MOVED MARKETS —", exp.date.date())
    print("=" * 60)
    print(write_note(exp))
    print("\nFactor contributions (pp of return):")
    print(contributions_table(exp).to_string(index=False))


if __name__ == "__main__":
    main()
