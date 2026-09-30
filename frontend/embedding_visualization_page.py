"""NiceGUI page layout for the embedding visualization."""

from nicegui import ui

from backend.embedding_visualization_controller import EmbeddingVisualizationController


TOGGLE_BUTTON_PROPERTIES = "spread unelevated no-caps color=red-1 text-color=red-10 toggle-color=red-10"


def create_embedding_visualization_page(controller: EmbeddingVisualizationController) -> None:
    """Build the browser page and connect controls to the visualization controller."""
    ui.colors(primary="#7f1d1d", secondary="#991b1b", accent="#b91c1c")
    with ui.column().classes("w-full max-w-7xl mx-auto gap-6 p-6"):
        ui.label("Database embedding visualization").classes("text-h4 text-red-10")
        ui.label("PCA and t-SNE always map embeddings to two dimensions. In 3D, harmfulness is the height.").classes(
            "text-body1 text-grey-7"
        )
        with ui.row().classes("w-full items-start gap-6 no-wrap"):
            with ui.card().classes("w-64 shrink-0 border border-red-100 bg-red-50"):
                ui.label("Controls").classes("text-h6 text-red-10")
                projection_toggle = ui.toggle(
                    ["PCA", "t-SNE"], value=controller.projection_method, on_change=lambda event: refresh_plot()
                ).props(TOGGLE_BUTTON_PROPERTIES)
                ui.label("Projection").classes("text-caption text-red-8")
                view_toggle = ui.toggle(
                    ["2D", "3D"], value=controller.view_dimension, on_change=lambda event: refresh_plot()
                ).props(TOGGLE_BUTTON_PROPERTIES)
                ui.label("View").classes("text-caption text-red-8")
                score_toggle = ui.toggle(
                    ["Prompt harmfulness", "Image harmfulness"],
                    value=controller.score_field,
                    on_change=lambda event: refresh_plot(),
                ).props(TOGGLE_BUTTON_PROPERTIES)
                ui.label("Colour / height").classes("text-caption text-red-8")
            with ui.card().classes("flex-grow min-w-0 border border-red-100"):
                plot = ui.plotly(controller.build_figure()).classes("w-full h-[70vh]")

    def refresh_plot() -> None:
        """Apply all selected values together and redraw the interactive Plotly chart."""
        controller.set_projection_method(projection_toggle.value)
        controller.set_view_dimension(view_toggle.value)
        controller.set_score_field(score_toggle.value)
        plot.update_figure(controller.build_figure())
