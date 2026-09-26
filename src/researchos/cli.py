"""Command line entry point.

``uv run researchos <command>``

Declared in pyproject.toml under [project.scripts], which installs a real
executable into the venv's bin/ directory. Not a convenience - it means the
tool works in Docker, CI, and cron the same way it works on your laptop.
"""

from __future__ import annotations

import argparse
import sys

from researchos import __version__


def cmd_migrate(_: argparse.Namespace) -> int:
    """Apply pending database migrations."""
    from researchos.db.migrate import main as migrate_main

    return migrate_main()


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the development server."""
    import uvicorn

    from researchos.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "researchos.main:create_app",
        factory=True,
        host=args.host or settings.app_host,
        port=args.port or settings.app_port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    """Verify configuration and dependency reachability. No side effects."""
    from researchos.config import get_settings
    from researchos.logging_config import setup_logging

    setup_logging()
    settings = get_settings()

    print(f"ResearchOS {__version__}")
    print(f"  postgres : {settings.database_url_safe}")
    print(f"  qdrant   : {settings.qdrant_url}")
    print(f"  embed    : {settings.embedding_model} ({settings.embedding_dim}d)")
    print(f"  rerank   : {settings.reranker_model}")
    print(f"  llm      : {' -> '.join(settings.gemini_models)}")
    print(f"  chunking : {settings.chunk_size_tokens}t / {settings.chunk_overlap_tokens}t overlap")
    print(f"  top_k    : {settings.retrieval_top_k} of {settings.retrieval_candidates} candidates")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchos", description=__doc__)
    parser.add_argument("--version", action="version", version=f"ResearchOS {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="apply pending database migrations").set_defaults(
        func=cmd_migrate
    )
    sub.add_parser("check", help="verify configuration, no side effects").set_defaults(
        func=cmd_check
    )

    serve = sub.add_parser("serve", help="run the development server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="auto-reload on file changes")
    serve.set_defaults(func=cmd_serve)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
