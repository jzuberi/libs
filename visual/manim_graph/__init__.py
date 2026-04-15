from manim import config
from .dispatcher import get_scene_class

def render_graph(graph_type, params, output_path):
    SceneClass = get_scene_class(graph_type)

    # Configure output
    config.output_file = output_path
    config.format = "mp4"
    config.write_to_movie = True

    # Instantiate + render
    scene = SceneClass(params)
    scene.render()

    return output_path
