"""CLI for indexing Parkinson's disease PubMed literature."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

try:
    from parkinson_ai.rag.pubmed_indexer import PubMedIndexer, build_default_pd_queries
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from parkinson_ai.rag.pubmed_indexer import PubMedIndexer, build_default_pd_queries


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Index Parkinson's disease PubMed abstracts.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Optional PubMed query. Repeat to provide multiple queries.",
    )
    parser.add_argument("--retmax", type=int, default=20, help="Maximum PMIDs per query.")
    parser.add_argument("--sort", default="relevance", help="PubMed sort mode.")
    parser.add_argument(
        "--show-pmids",
        action="store_true",
        help="Include the indexed PMID list in the printed summary.",
    )
    return parser.parse_args()


async def _run_indexer(queries: list[str] | None, *, retmax: int, sort: str) -> dict[str, Any]:
    """Run the async PubMed indexing workflow and return a JSON-safe summary."""

    indexer = PubMedIndexer()
    summary = await indexer.fetch_and_index(queries or build_default_pd_queries(), retmax=retmax, sort=sort)
    return {
        "query_count": summary.query_count,
        "article_count": summary.article_count,
        "chunk_count": summary.chunk_count,
        "pmids": summary.pmids,
    }


def main() -> None:
    """Index PD literature and print a compact summary."""

    args = parse_args()
    summary = asyncio.run(_run_indexer(args.queries, retmax=args.retmax, sort=args.sort))
    if not args.show_pmids:
        summary = {key: value for key, value in summary.items() if key != "pmids"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
