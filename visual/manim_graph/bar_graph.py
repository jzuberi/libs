# manim_graph/bar_graph.py
from manim import *
from .base import GraphScene
import random
from manim_graph.utils import compute_axis_scale


class BarGraphScene(GraphScene):
    def construct(self):
        p = self.params

        # -----------------------------
        # Camera settings
        # -----------------------------
        # Use width/height as the *pixel* resolution
        pixel_w = p.get("width", 2160)
        pixel_h = p.get("height", 3840)

        self.camera.pixel_width = pixel_w
        self.camera.pixel_height = pixel_h
        self.camera.frame_rate = p.get("fps", 60)

        # Make the logical frame match the pixel aspect ratio
        # Start from Manim's default frame_height, adjust frame_width.
        base_frame_h = config.frame_height
        aspect = pixel_w / pixel_h if pixel_h != 0 else 1.0
        self.camera.frame_height = base_frame_h
        self.camera.frame_width = base_frame_h * aspect

        # Colors
        BG_COLOR = p.get("bg_color", "#000000")
        AXIS_COLOR = p.get("axis_color", WHITE)
        BAR_COLOR = p.get("bar_color", "#8888ff")

        DATA_X = p["data_x"]
        DATA_Y = p["data_y"]

        self.camera.background_color = BG_COLOR

        # -----------------------------
        # Axis titles
        # -----------------------------
        x_axis_title = p.get("x_axis_title", "")
        y_axis_title = p.get("y_axis_title", "")

        axis_title_font = p.get("axis_title_font", "Helvetica Neue")
        axis_title_font_size = p.get("axis_title_font_size", 32)

        # -----------------------------
        # Label styling
        # -----------------------------
        x_label_rotation = p.get("x_label_rotation", 90)
        x_label_font_size = p.get("x_label_font_size", 28)
        y_label_font_size = p.get("y_label_font_size", 20)

        # -----------------------------
        # Bar styling
        # -----------------------------
        bar_width = p.get("bar_width", 0.75)
        texture_layers = p.get("texture_layers", 7)

        # -----------------------------
        # Animation settings
        # -----------------------------
        anim = p.get("animation", {})
        grow_lag = anim.get("grow_lag", 0.1)
        grow_time = anim.get("grow_time", 0.5)
        hold_time = anim.get("hold_time", 4)
        collapse_lag = anim.get("collapse_lag", 0.1)

        # -----------------------------
        # Compute y-axis range
        # -----------------------------
        y_min, y_max, y_step = compute_axis_scale(DATA_Y, p)

        VISIBLE_DATA_Y = [y if y > 0 else 0.001 for y in DATA_Y]

        # -----------------------------
        # Build chart
        # -----------------------------
        chart = BarChart(
            values=VISIBLE_DATA_Y,
            y_range=[y_min, y_max, y_step],
            bar_width=bar_width,
        )

        chart.axes.set_color(AXIS_COLOR)

        # -----------------------------
        # Bar styling + texture layers
        # -----------------------------
        texture_registry = {}

        for bar, y in zip(chart.bars, DATA_Y):
            bar.set_fill(BAR_COLOR if y > 0 else GRAY)
            bar.set_stroke(color=AXIS_COLOR, width=1, opacity=0.6)

            layers = VGroup()
            for _ in range(texture_layers):
                band = Rectangle(
                    width=bar.width * (0.85 + random.random() * 0.3),
                    height=0.001,
                    fill_color=interpolate_color(GRAY, BLACK, random.random() * 0.3),
                    fill_opacity=0.05 + random.random() * 0.08,
                    stroke_width=0,
                )
                band.shift(RIGHT * (random.random() - 0.5) * bar.width * 0.1)
                band.move_to(bar.get_bottom() + UP * 0.0005)
                band.set_z_index(bar.z_index + 1)
                layers.add(band)

            texture_registry[bar] = layers
            bar.add(*layers)

        # -----------------------------
        # Y-axis title
        # -----------------------------
        if y_axis_title:
            y_title = Text(
                y_axis_title,
                font=axis_title_font,
                font_size=axis_title_font_size,
                color=AXIS_COLOR
            )
            y_title.rotate(90 * DEGREES)
            y_title.next_to(chart.axes[1], LEFT, buff=0.3)
            y_title.set_stroke(width=1.2, color=AXIS_COLOR, opacity=0.8)
        else:
            y_title = VGroup()

        # -----------------------------
        # X-axis labels
        # -----------------------------
        x_axis = chart.axes[0]

        x_labels = VGroup()
        for bar, x_val in zip(chart.bars, DATA_X):
            label = Text(str(x_val), font_size=x_label_font_size, color=AXIS_COLOR)
            label.rotate(x_label_rotation * DEGREES)

            tick_x = bar.get_x()
            label_height = label.height
            offset = label_height * 0.7
            label.move_to([tick_x, x_axis.get_y() - offset, 0])

            x_labels.add(label)

        if x_axis_title:
            x_title = Text(
                x_axis_title,
                font=axis_title_font,
                font_size=axis_title_font_size,
                color=AXIS_COLOR
            )
            x_title.next_to(chart.axes[0], DOWN, buff=0.4)
            x_title.set_stroke(width=1.2, color=AXIS_COLOR, opacity=0.8)
        else:
            x_title = VGroup()

        # Group everything that should scale together
        graph_group = VGroup(chart, x_labels, y_title, x_title)

        # -----------------------------
        # Pixel → unit scaling (now using camera frame)
        # -----------------------------
        FRAME_W_UNITS = self.camera.frame_width
        FRAME_H_UNITS = self.camera.frame_height

        px_to_unit_x = FRAME_W_UNITS / pixel_w
        px_to_unit_y = FRAME_H_UNITS / pixel_h

        target_w_units = p["width"] * px_to_unit_x
        target_h_units = p["height"] * px_to_unit_y

        scale_factor = min(
            target_w_units / graph_group.width,
            target_h_units / graph_group.height,
        )

        graph_group.scale(scale_factor)
        graph_group.move_to(ORIGIN)

        # -----------------------------
        # Animation: grow bars
        # -----------------------------
        self.play(
            Create(chart.axes[0]),
            Create(chart.axes[1]),
            FadeIn(x_labels, shift=DOWN * 0.2),
            FadeIn(y_title, shift=LEFT * 0.1),
            run_time=grow_time,
        )

        grow_anims = []
        for bar in chart.bars:
            layers = texture_registry[bar]
            grow_anims.append(
                AnimationGroup(
                    GrowFromEdge(bar, DOWN),
                    *[GrowFromEdge(layer, DOWN) for layer in layers],
                    lag_ratio=0.0,
                )
            )

        self.play(LaggedStart(*grow_anims, lag_ratio=grow_lag))

        self.wait(hold_time)

        # -----------------------------
        # Reverse animation
        # -----------------------------
        reverse_anims = []
        for bar in chart.bars:
            layers = texture_registry[bar]

            collapsed_bar = bar.copy()
            collapsed_bar.stretch_to_fit_height(0.001)
            collapsed_bar.move_to(bar.get_bottom(), DOWN)

            collapsed_layers = VGroup()
            for layer in layers:
                collapsed_layer = layer.copy()
                collapsed_layer.stretch_to_fit_height(0.001)
                collapsed_layer.move_to(layer.get_bottom(), DOWN)
                collapsed_layers.add(collapsed_layer)

            reverse_anims.append(
                AnimationGroup(
                    Transform(bar, collapsed_bar),
                    *[
                        Transform(layer, cl)
                        for layer, cl in zip(layers, collapsed_layers)
                    ],
                    lag_ratio=0.0,
                )
            )

        self.play(LaggedStart(*reverse_anims, lag_ratio=collapse_lag))
        self.play(
            FadeOut(chart),
            FadeOut(x_labels),
            FadeOut(y_title),
            run_time=0.2
        )

        self.wait()
