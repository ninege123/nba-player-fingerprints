"""Command-line entry points for project pipelines."""

from __future__ import annotations

import argparse

from nba_fingerprints.pipelines.player_season_exports import export_player_season_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA player fingerprint utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-player-season", help="Export processed player-season tables")
    export_parser.add_argument("--season", required=True, help="NBA season label, for example 2023-24")
    export_parser.add_argument("--min-minutes", type=float, default=500.0, help="Minimum season minutes to include")
    export_parser.add_argument("--top-n", type=int, default=5, help="Nearest neighbors per player")
    export_parser.add_argument("--output-dir", default="data/processed", help="Directory for processed outputs")
    export_parser.add_argument("--raw-cache-dir", default="data/raw", help="Directory for raw API cache files")
    export_parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv", help="Output and cache file format")
    export_parser.add_argument("--refresh-cache", action="store_true", help="Fetch from the API even if raw cache exists")

    args = parser.parse_args()

    if args.command == "export-player-season":
        paths = export_player_season_tables(
            season=args.season,
            min_minutes=args.min_minutes,
            top_n=args.top_n,
            output_dir=args.output_dir,
            raw_cache_dir=args.raw_cache_dir,
            file_format=args.file_format,
            use_cache=not args.refresh_cache,
        )
        print(f"features: {paths.features}")
        print(f"fingerprints: {paths.fingerprints}")
        print(f"neighbors: {paths.neighbors}")
        print(f"position_references: {paths.position_references}")
        print(f"position_scores: {paths.position_scores}")
        print(f"archetype_references: {paths.archetype_references}")
        print(f"archetype_scores: {paths.archetype_scores}")
        print(f"archetype_explanations: {paths.archetype_explanations}")


if __name__ == "__main__":
    main()
