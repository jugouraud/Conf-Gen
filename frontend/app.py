"""Launch the NiceGUI database embedding visualizer."""

import argparse
from pathlib import Path

from nicegui import ui

from backend.storage.fairness import DEFAULT_DATABASE_PATH
from frontend.page import create_embedding_visualization_page
from frontend.plots import EmbeddingVisualizationController, load_visualization_data


def main() -> None:
    """Load visualization data, build the browser page, and start NiceGUI."""
    parser = argparse.ArgumentParser(description="Interactively visualize database inner embeddings in a browser.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database to visualize.")
    parser.add_argument("--port", type=int, default=8080, help="Local web server port.")
    args = parser.parse_args()

    controller = EmbeddingVisualizationController(load_visualization_data(args.database))
    create_embedding_visualization_page(controller)
    ui.run(title="Embedding validation view", port=args.port)


if __name__ in {"__main__", "__mp_main__"}:
    main()
