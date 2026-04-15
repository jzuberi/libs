# manim_graph/base.py
from manim import Scene

class GraphScene(Scene):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
