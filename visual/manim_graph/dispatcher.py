# manim_graph/dispatcher.py
from .bar_graph import BarGraphScene
# from .line_graph import LineGraphScene
# from .pie_chart import PieChartScene
# etc.

GRAPH_TYPES = {
    "bar": BarGraphScene,
    # "line": LineGraphScene,
    # "pie": PieChartScene,
}


def get_scene_class(graph_type: str):
    try:
        return GRAPH_TYPES[graph_type]
    except KeyError:
        raise ValueError(f"Unknown graph type: {graph_type!r}. Available: {list(GRAPH_TYPES.keys())}")
