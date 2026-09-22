import sys
from pathlib import Path

import manim_graph
from upscale_creative import upscale_image, upscale_video

from PIL import Image, ImageDraw, ImageFont
import cairosvg, io
import numpy as np

from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ImageClip, ColorClip
import xml.etree.ElementTree as ET




class SlideAnimator:
    def __init__(self, png_path, slot_geometries, audio_clip):
        self.png_path = png_path
        self.slot_geometries = slot_geometries
        self.audio_clip = audio_clip
        self.duration = audio_clip.duration

        print("\n[SlideAnimator.__init__]")
        print("  png_path:", self.png_path)
        print("  duration (audio):", self.duration)
        print("  slots keys:", list(self.slot_geometries.keys()))

    # ------------------------------------------------------------
    # Correct fade logic for RGBA clips
    # ------------------------------------------------------------
    def apply_animation_relative(self, clip, anim):
        print("\n[apply_animation_relative] ENTER")
    
        # Ensure mask exists
        mask = clip.mask
        if mask is None:
            mask = ImageClip(np.ones((clip.h, clip.w)) * 255).set_duration(self.duration)
            clip = clip.set_mask(mask)
    
        # ------------------------------------------------------------
        # Fade-in / fade-out (your existing logic)
        # ------------------------------------------------------------
        if "in" in anim:
            a = anim["in"]
            mask = mask.fadein(a.get("duration", 0.4))
    
        if "out" in anim:
            a = anim["out"]
            mask = mask.fadeout(a.get("duration", 0.6))
    
        # ------------------------------------------------------------
        # ⭐ NEW: Pulse animation
        # ------------------------------------------------------------
        if anim.get("type") == "pulse":
            freq = anim.get("frequency", 8.0)      # much faster
            min_alpha = anim.get("min", 0.7)       # barely changes
            max_alpha = anim.get("max", 1.0)
        
            def pulse_alpha(gf, t):
                # triangle wave (sharper, more glitchy)
                tri = abs((t * freq) % 2 - 1)
        
                # small random jitter
                jitter = np.random.uniform(-0.05, 0.05)
        
                # combine
                alpha = min_alpha + (max_alpha - min_alpha) * tri + jitter
        
                # clamp
                alpha = max(0.0, min(1.0, alpha))
        
                # apply to full mask frame
                frame = gf(t)
                return frame * alpha
        
            mask = mask.fl(pulse_alpha)


    
        clip = clip.set_mask(mask)
        print("[apply_animation_relative] EXIT")
        return clip


    # ------------------------------------------------------------
    # Build final composite clip
    # ------------------------------------------------------------
    def build(self):
        print("\n[SlideAnimator.build] BEGIN")
        print("  duration:", self.duration)

        # Base PNG
        base = (
            ImageClip(self.png_path)
            .set_duration(self.duration)
            .set_start(0)
            .set_audio(self.audio_clip)
        )

        overlays = []

        for name, geo in self.slot_geometries.items():
            print("\n[SlideAnimator.build] slot:", name)
            print("  geo:", geo)

            if geo["type"] != "svg":
                continue

            png_bytes = cairosvg.svg2png(
                url=geo["path"],
                output_width=geo["width"],
                output_height=geo["height"]
            )
            svg_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

            clip = ImageClip(np.array(svg_img)).set_position((geo["x"], geo["y"]))
            clip = clip.set_duration(self.duration).set_start(0)

            anim = geo.get("animate")
            if anim:
                clip = self.apply_animation_relative(clip, anim)

            overlays.append(clip)

        print("\n[SlideAnimator.build] overlays count:", len(overlays))

        comp = CompositeVideoClip([base] + overlays)
        print("[SlideAnimator.build] END")

        return comp


def apply_animation_relative(self, clip, anim):
    # Fade-in
    if "in" in anim:
        a = anim["in"]
        start_at = a.get("start_at", "start")

        if start_at == "start":
            clip = clip.set_start(0).fadein(a["duration"])
        elif isinstance(start_at, (int, float)):
            clip = clip.set_start(start_at).fadein(a["duration"])

    # Fade-out
    if "out" in anim:
        a = anim["out"]
        start_at = a.get("start_at", "end")

        if start_at == "end":
            start_time = self.duration - a["duration"]
            clip = clip.set_start(start_time).fadeout(a["duration"])
        elif isinstance(start_at, (int, float)):
            clip = clip.set_start(start_at).fadeout(a["duration"])

    return clip



def mask_svg_to_circle(input_path, rgb=(0,0,0), output_path=None):
    """
    Apply a perfectly smooth circular fade near the edges of an SVG,
    fading toward the given RGB tuple. The center remains fully original.
    """

    tree = ET.parse(input_path)
    root = tree.getroot()

    ET.register_namespace("", "http://www.w3.org/2000/svg")

    # Extract viewBox
    viewBox = root.get("viewBox")
    if not viewBox:
        raise ValueError("SVG must have a viewBox to compute circle mask.")

    min_x, min_y, width, height = map(float, viewBox.split())

    # Slightly smaller circle to avoid clipping
    base_r = min(width, height) / 2
    r = base_r * 0.90   # <-- shrink circle a bit more

    cx = min_x + width / 2
    cy = min_y + height / 2

    # Ensure <defs> exists
    defs = root.find("{http://www.w3.org/2000/svg}defs")
    if defs is None:
        defs = ET.SubElement(root, "defs")

    # ---------------------------------------------------------
    # Smooth feather mask using Gaussian blur + gamma correction
    # ---------------------------------------------------------
    mask = ET.SubElement(defs, "mask", {"id": "circleFadeMask"})

    # Hard circle (will be blurred)
    hard_circle = ET.SubElement(
        mask,
        "circle",
        {
            "cx": str(cx),
            "cy": str(cy),
            "r": str(r),
            "fill": "white"
        }
    )

    # Feather filter with expanded region
    filt = ET.SubElement(
        defs,
        "filter",
        {
            "id": "featherFilter",
            "x": "-20%",
            "y": "-20%",
            "width": "140%",
            "height": "140%",
            "color-interpolation-filters": "sRGB"
        }
    )

    # Larger blur radius → smoother fade
    ET.SubElement(
        filt,
        "feGaussianBlur",
        {
            "in": "SourceGraphic",
            "stdDeviation": str(r * 0.20)  # smoother feather
        }
    )

    # Gamma correction → removes banding
    comp = ET.SubElement(filt, "feComponentTransfer")
    ET.SubElement(comp, "feFuncA", {
        "type": "gamma",
        "exponent": "1.8",   # smoother falloff curve
        "amplitude": "1"
    })

    # Apply filter to circle
    hard_circle.set("filter", "url(#featherFilter)")

    # ---------------------------------------------------------
    # Background rectangle filled with fade color
    # ---------------------------------------------------------
    fade_color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    bg_rect = ET.Element(
        "rect",
        {
            "x": str(min_x),
            "y": str(min_y),
            "width": str(width),
            "height": str(height),
            "fill": fade_color
        }
    )

    # ---------------------------------------------------------
    # Wrap original content in masked group
    # ---------------------------------------------------------
    masked_group = ET.Element("g", {"mask": "url(#circleFadeMask)"})

    for child in list(root):
        if child.tag.endswith("defs"):
            continue
        root.remove(child)
        masked_group.append(child)

    root.append(bg_rect)
    root.append(masked_group)

    # Save output
    if output_path is None:
        output_path = input_path.replace(".svg", "_circle_fade.svg")

    tree.write(output_path)
    return output_path






def render_graph_slot(*, geom, payload, output_path):
    """
    Slot renderer that mirrors GraphSlide's graph rendering behavior.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "width": int(geom["width"]),
        "height": int(geom["height"]),
        "data_x": payload["data_x"],
        "data_y": payload["data_y"],
        "x_axis_title": payload.get("x_axis_title", ""),
        "y_axis_title": payload.get("y_axis_title", ""),
        "theme": payload.get("theme", "dark"),
    }

    # Mirror GraphSlide: merge graph_options if present
    params.update(payload.get("graph_options", {}))
    
    print('---')
    print(out_path)
    print(params)

    manim_graph.render_graph(
        graph_type=payload["graph_type"],
        params=params,
        output_path=str(out_path)
    )


SLOT_RENDERERS = {}

SLOT_RENDERERS["graph"] = render_graph_slot

def render_raw_video(geom, payload, output_path):
    clip = VideoFileClip(payload["path"])
    clip.write_videofile(output_path, codec="libx264", audio=False)

SLOT_RENDERERS["raw_video"] = render_raw_video


def normalize_color(c):
    if isinstance(c, list):
        return tuple(c)
    return c


class SlideRenderer:
    
    def __init__(self, layout_spec, canvas_size, scale=1):
        self.spec = layout_spec
        self.blocks = layout_spec["blocks"]

        # -----------------------------------
        # Logical canvas (unscaled)
        # canvas_size = (width, height or None)
        # -----------------------------------
        self.logical_canvas = canvas_size
        self.scale = scale

        w, h = canvas_size

        # Width must always be concrete
        if w is None:
            raise ValueError("SlideRenderer requires a concrete canvas width.")

        # Height may be None (measurement mode)
        if h is None:
            # Store width scaled, height None
            self.canvas_size = (int(w * scale), None)
        else:
            # Normal fixed-size mode
            self.canvas_size = (int(w * scale), int(h * scale))

        # -----------------------------
        # Font registry
        # -----------------------------
        self.fonts = layout_spec.get("fonts", {})
        self.default_font = self.fonts.get("courier")
        self.default_bold = self.fonts.get("courier_bold")

        # -----------------------------
        # Base layout settings
        # (these will be scaled by global g)
        # -----------------------------
        self.base_padding = layout_spec.get("padding", 100)
        self.base_max_width_factor = layout_spec.get("max_width_factor", 1)

        self.default_color = normalize_color(
            layout_spec.get("default_text_color", [255, 255, 255])
        )
        self.default_line_gap = layout_spec.get("default_line_gap", 10)


        # -----------------------------
        # Frame settings (also scaled by g)
        # -----------------------------
        frame = layout_spec.get("frame")
        if frame:
            self.base_frame = {
                "color": normalize_color(frame.get("color")),
                "thickness": frame.get("thickness", 8),
                "radius": frame.get("radius", 50),
                "inset": frame.get("inset", 40),
            }
        else:
            self.base_frame = None

        # -----------------------------
        # Global scale factor g
        # (starts at 1.0, shrinks until layout fits)
        # -----------------------------
        self.global_scale = 1.0

        # text minimum size clamp (E2)
        self.min_text_px = 1
        
    # ------------------------------------------------------------
    # Font loader
    # ------------------------------------------------------------
    def load_font(self, font_name, size):
        path = self.fonts.get(font_name)
        if path is None:
            if font_name == "courier_bold":
                path = self.default_bold
            else:
                path = self.default_font
        return ImageFont.truetype(path, size)
        
        
    # ------------------------------------------------------------
    # Apply global scale g to all layout parameters
    # ------------------------------------------------------------
    def scale_layout(self, g):
        # scaled padding
        padding = self.base_padding * g

        # scaled frame
        if self.base_frame:
            frame = {
                "color": self.base_frame["color"],
                "thickness": max(1, self.base_frame["thickness"] * g),
                "radius": self.base_frame["radius"] * g,
                "inset": self.base_frame["inset"] * g,
            }
        else:
            frame = None

        return padding, frame
    
    # ------------------------------------------------------------
    # Text wrapping
    # ------------------------------------------------------------
    def wrap_text(self, text, font, max_width, draw):
        words = text.split()
        lines = []
        current = ""

        for w in words:
            trial = (current + " " + w).strip()
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = w

        if current:
            lines.append(current)

        return lines

    # ------------------------------------------------------------
    # Measure all blocks at scale g
    # ------------------------------------------------------------
    def measure_blocks(self, draw, g):
        measured = []

        # scaled padding + frame
        padding, frame = self.scale_layout(g)

        # ------------------------------------------------------------
        # SAFE AREA (width + height)
        # ------------------------------------------------------------
        safe_left = padding
        safe_right = self.canvas_size[0] - padding
        safe_top = padding
        safe_bottom = self.canvas_size[1] - padding

        if frame:
            inset = frame["inset"] + frame["thickness"]
            safe_left += inset
            safe_right -= inset
            safe_top += inset
            safe_bottom -= inset

        safe_width = max(1, safe_right - safe_left)
        safe_height = max(1, safe_bottom - safe_top)

        # ------------------------------------------------------------
        # MAX TEXT WIDTH (inside safe area)
        # ------------------------------------------------------------
        max_width = int(safe_width * self.base_max_width_factor)

        for idx, block in enumerate(self.blocks):
            btype = block["type"]

            # -----------------------------
            # TEXT
            # -----------------------------
            if btype == "text":
                base_size = block.get("size", 60)
                size = max(self.min_text_px, base_size * g)

                font_name = block.get("font", "courier")
                line_gap = block.get("line_gap", self.default_line_gap) * g

                font = self.load_font(font_name, int(max(1, size)))
                lines = self.wrap_text(block["text"], font, max_width, draw)

                base_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
                line_h = base_h + line_gap
                total_h = len(lines) * line_h

                measured.append({
                    "type": "text",
                    "index": idx,
                    "lines": lines,
                    "font": font,
                    "line_h": line_h,
                    "height": total_h,
                    "color": normalize_color(block.get("color", self.default_color)),
                    "gap_after": block.get("gap_after", 20) * g,
                    "animate": block.get("animate")
                })
            elif btype == "text_and_background":
                base_size = block.get("size", 60)
                size = max(self.min_text_px, base_size * g)

                font_name = block.get("font", "courier")
                font = self.load_font(font_name, int(size))

                # wrap text
                lines = self.wrap_text(block["text"], font, max_width, draw)

                # measure text height
                base_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
                line_gap = block.get("line_gap", self.default_line_gap) * g
                line_h = base_h + line_gap
                text_height = len(lines) * line_h

                # background padding (scaled)
                pad_top = block.get("padding_top", 40) * g
                pad_bottom = block.get("padding_bottom", 40) * g

                total_height = pad_top + text_height + pad_bottom

                measured.append({
                    "type": "text_and_background",
                    "lines": lines,
                    "font": font,
                    "line_h": line_h,
                    "height": total_height,
                    "bg_color": normalize_color(block.get("bg_color", [0,0,0])),
                    "pad_top": pad_top,
                    "pad_bottom": pad_bottom,
                    "color": normalize_color(block.get("color", self.default_color)),
                    "gap_after": block.get("gap_after", 20) * g
                })

                
            # -----------------------------
            # RICH TEXT
            # -----------------------------
            # ------------------------------------------------------------
            # RICH TEXT BLOCK (with wrapping)
            # ------------------------------------------------------------
            elif btype == "rich_text":
                spans = block["spans"]
                
                lines = []
                current_line = []
                current_width = 0
                max_ascent = 0
                max_descent = 0

                def push_line():
                    if current_line:
                        line_height = max_ascent + max_descent
                        lines.append({
                            "spans": current_line.copy(),
                            "width": current_width,
                            "height": line_height,
                        })

                for sp in spans:
                    size = max(self.min_text_px, sp["size"] * g)
                    font = self.load_font(sp["font"], int(size))

                    words = sp["text"].split(" ")
                    for i, word in enumerate(words):
                        token = word + (" " if i < len(words) - 1 else "")
                        w = draw.textlength(token, font=font)

                        if current_line and (current_width + w > max_width):
                            push_line()
                            current_line = []
                            current_width = 0
                            max_ascent = 0
                            max_descent = 0

                        current_line.append({
                            "text": token,
                            "font": font,
                            "width": w,
                            "height": font.getbbox(token)[3] - font.getbbox(token)[1],
                            "color": sp.get("color", [255,255,255]),
                            "ascent": font.getmetrics()[0],
                            "descent": font.getmetrics()[1],
                        })

                        current_width += w
                        max_ascent = max(max_ascent, font.getmetrics()[0])
                        max_descent = max(max_descent, font.getmetrics()[1])


                push_line()

                total_height = sum(line["height"] for line in lines)
                max_line_width = max((line["width"] for line in lines), default=0)

                measured.append({
                    "type": "rich_text",
                    "lines": lines,
                    "width": max_line_width,
                    "height": total_height,
                    "gap_after": block.get("gap_after", 20) * g,
                    "animate": block.get("animate")
                })



            # -----------------------------
            # SVG
            # -----------------------------
            elif btype == "svg":
                scale = block.get("scale", 0.1) * g
                target_w = max(1, int(self.canvas_size[0] * scale))

                measured.append({
                    "type": "svg",
                    "index": idx,
                    "path": block["path"],
                    "width": target_w,
                    "height": target_w,
                    "gap_after": block.get("gap_after", 20) * g,
                    "animate": block.get("animate")
                })


            # -----------------------------
            # DIVIDER
            # -----------------------------
            elif btype == "divider":
                thickness = max(1, block.get("thickness", 4) * g)

                measured.append({
                    "type": "divider",
                    "index": idx,
                    "width_factor": block.get("width_factor", 0.7),
                    "thickness": thickness,
                    "height": thickness,
                    "color": normalize_color(block.get("color", self.default_color)),
                    "gap_after": block.get("gap_after", 20) * g
                })

            # -----------------------------
            # SPACER
            # -----------------------------
            elif btype == "spacer":
                measured.append({
                    "type": "spacer",
                    "index": idx,
                    "height": block.get("height", 40) * g,
                    "gap_after": 0
                })

            # -----------------------------
            # SLOT
            # -----------------------------
            elif btype == "slot":
                height = block.get("height", 400) * g
                measured.append({
                    "type": "slot",
                    "index": idx,
                    "slot_name": block["slot_name"],
                    "height": height,
                    "gap_after": block.get("gap_after", 20) * g
                })

        return measured, padding, frame, safe_height, safe_left, safe_right

    # ------------------------------------------------------------
    # Compute total vertical height of measured blocks
    # ------------------------------------------------------------
    def compute_total_height(self, measured):
        return sum(m["height"] + m.get("gap_after", 0) for m in measured)

    # ------------------------------------------------------------
    # Global scaling loop (U1): scale UP first, then DOWN if needed
    # with width constraints for text and rich_text
    # ------------------------------------------------------------
    def global_scale_pass(self, draw):
        

        # --- helper: compute max line width across all blocks ---
        def compute_max_line_width(measured_blocks):
            max_w = 0
            for m in measured_blocks:
                t = m["type"]
                if t == "text":
                    max_w = max(max_w, m.get("width", 0))
                elif t == "rich_text":
                    for line in m["lines"]:
                        max_w = max(max_w, line["width"])
            return max_w

        # --- 1) Measure at g = 1.0 ---
        
        g = 1.0
        measured, padding, frame, safe_height, safe_left, safe_right = self.measure_blocks(draw, g)
        drawable_height = safe_height

        total_h = self.compute_total_height(measured)

        # ------------------------------------------------------------
        # SAFE AREA WIDTH (must match measure_blocks + draw_blocks)
        # ------------------------------------------------------------

        safe_width = safe_right - safe_left
        raw_max_width = int(safe_width * self.base_max_width_factor)

        max_line_width = compute_max_line_width(measured)

        # --- 2) Scale UP until ~95% height fill, but only if width is safe ---
        target_fill = 0.95 * drawable_height

        if total_h < target_fill and max_line_width <= raw_max_width:
            for _ in range(200):  # safety cap
                g *= 1.05  # scale UP
                measured, padding, frame, safe_height, safe_left, safe_right = self.measure_blocks(draw, g)
                total_h = self.compute_total_height(measured)
                max_line_width = compute_max_line_width(measured)

                # stop if height target reached OR width exceeded
                if total_h >= target_fill or max_line_width > raw_max_width:
                    break

        # --- 3) If height OR width overflow, scale DOWN ---
        if total_h > drawable_height or max_line_width > raw_max_width:
            for _ in range(200):
                g *= 0.95  # scale DOWN
                measured, padding, frame, safe_height, safe_left, safe_right = self.measure_blocks(draw, g)
                total_h = self.compute_total_height(measured)
                max_line_width = compute_max_line_width(measured)

                # stop when BOTH height and width fit
                if total_h <= drawable_height and max_line_width <= raw_max_width:
                    break

        # --- 4) Save final scale ---
        self.global_scale = g
        return measured, padding, frame

    # ------------------------------------------------------------
    # Compute vertical positions for measured blocks
    # ------------------------------------------------------------
    def compute_positions(self, measured, padding, frame):
        safe_top = padding
        safe_bottom = self.canvas_size[1] - padding

        if frame:
            safe_top += frame["inset"] + frame["thickness"]
            safe_bottom -= frame["inset"] + frame["thickness"]

        drawable_height = max(1, safe_bottom - safe_top)

        total_h = self.compute_total_height(measured)

        # If content still exceeds drawable height (rare with D2/E2),
        # compress gaps proportionally.
        if total_h > drawable_height:
            overflow = total_h - drawable_height
            gap_blocks = [m for m in measured if m.get("gap_after", 0) > 0]
            total_gaps = sum(m.get("gap_after", 0) for m in gap_blocks)

            if total_gaps > 0:
                for m in gap_blocks:
                    g = m["gap_after"]
                    shrink = overflow * (g / total_gaps)
                    m["gap_after"] = max(0, g - shrink)

            total_h = self.compute_total_height(measured)

        # Now compute positions
        y = safe_top
        positions = []
        for m in measured:
            positions.append(y)
            y += m["height"] + m.get("gap_after", 0)

        return positions

    # ------------------------------------------------------------
    # Draw all blocks at their computed positions
    # ------------------------------------------------------------
    def draw_blocks(self, draw, img, measured, positions, padding, frame):
        W, H = self.canvas_size
        self.slot_geometries = {}

        for m, y in zip(measured, positions):
            btype = m["type"]

            # -----------------------------
            # TEXT
            # -----------------------------
            if btype == "text":
                safe_left = padding
                safe_right = W - padding
                if frame:
                    safe_left += frame["inset"] + frame["thickness"]
                    safe_right -= frame["inset"] + frame["thickness"]

                safe_width = safe_right - safe_left

                for line in m["lines"]:
                    w = draw.textlength(line, font=m["font"])
                    x = safe_left + (safe_width - w) // 2
                    draw.text((x, y), line, font=m["font"], fill=m["color"])
                    y += m["line_h"]

            elif btype == "text_and_background":
                x1 = 0
                x2 = W

                draw.rectangle(
                    [x1, y, x2, y + m["height"]],
                    fill=m["bg_color"]
                )

                safe_width = x2 - x1
                ty = y + m["pad_top"]

                for line in m["lines"]:
                    w = draw.textlength(line, font=m["font"])
                    tx = x1 + (safe_width - w) // 2
                    draw.text((tx, ty), line, font=m["font"], fill=m["color"])
                    ty += m["line_h"]

            # -----------------------------
            # RICH TEXT (wrapped)
            # -----------------------------
            elif btype == "rich_text":
                for line in m["lines"]:
                    safe_left = padding
                    safe_right = W - padding
                    if frame:
                        safe_left += frame["inset"] + frame["thickness"]
                        safe_right -= frame["inset"] + frame["thickness"]

                    safe_width = safe_right - safe_left
                    x = safe_left + (safe_width - line["width"]) // 2

                    for sp in line["spans"]:
                        draw.text(
                            (x, y),
                            sp["text"],
                            font=sp["font"],
                            fill=tuple(sp["color"])
                        )
                        x += sp["width"]

                    y += line["height"]

            # -----------------------------
            # SVG
            # -----------------------------
            elif btype == "svg":
                print("\n[DEBUG] SVG block:")
                print("  path:", m["path"])
                print("  requested width:", m["width"])
                print("  requested height:", m["height"])
                print("  scale:", m.get("scale"))

                # If animated → DO NOT draw into PNG
                if m.get("animate"):
                    x = max(0, (W - m["width"]) // 2)
                    print("  computed x:", x)
                    print("  computed y:", int(y))

                    self.slot_geometries[f"svg_{m['index']}"] = {
                        "type": "svg",
                        "x": x,
                        "y": int(y),
                        "width": m["width"],
                        "height": m["height"],
                        "path": m["path"],
                        "animate": m.get("animate"),
                    }

                    continue

                # Otherwise → STATIC SVG → draw normally
                png_bytes = cairosvg.svg2png(
                    url=m["path"],
                    output_width=m["width"],
                    output_height=m["height"]
                )
                svg_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

                x = (W - m["width"]) // 2
                img.paste(svg_img, (x, int(y)), svg_img)




            # -----------------------------
            # DIVIDER
            # -----------------------------
            elif btype == "divider":
                line_w = int(W * m["width_factor"])
                x1 = (W - line_w) // 2
                x2 = x1 + line_w
                draw.line((x1, y, x2, y), fill=m["color"], width=int(m["thickness"]))

            # -----------------------------
            # SPACER
            # -----------------------------
            elif btype == "spacer":
                pass

            # -----------------------------
            # SLOT
            # -----------------------------
            elif btype == "slot":
                content_width = W - 2 * padding
                x = padding

                self.slot_geometries[m["slot_name"]] = {
                    "x": x,
                    "y": y,
                    "width": content_width,
                    "height": m["height"],
                }

    # ------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------
    def render(self, output_path=None):
        
        bg = normalize_color(self.spec.get("background_color", [0, 0, 0]))

        # create canvas (fixed size)
        img = Image.new("RGB", self.canvas_size, bg)
        draw = ImageDraw.Draw(img)

        # global scaling pass
        measured, padding, frame = self.global_scale_pass(draw)
        

        # compute positions
        positions = self.compute_positions(measured, padding, frame)

        # draw frame
        if frame:
            
            inset = frame["inset"]
            thickness = frame["thickness"]
            radius = frame["radius"]
            color = frame["color"]

            x1 = inset
            y1 = inset
            x2 = self.canvas_size[0] - inset
            y2 = self.canvas_size[1] - inset

            draw.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=radius,
                outline=color,
                width=int(thickness)
            )

        # draw blocks
        self.draw_blocks(draw, img, measured, positions, padding, frame)

        # save if requested
        if output_path:
            img.save(output_path)

        return img
    

    # ------------------------------------------------------------
    # Render + return slot geometries
    # ------------------------------------------------------------
    def render_with_slots(self, output_path=None):
        img = self.render(output_path=output_path)
        return img, getattr(self, "slot_geometries", {})




def composite_slide_with_slots(
    *,
    static_png,
    slot_videos,        # dict: slot_name → video_path
    slot_geometries,    # dict: slot_name → {x, y, width, height}
    output_path
):
    """
    Composites the static PNG with one or more slot videos using MoviePy.
    Each slot video is resized and positioned according to its geometry.
    """

    # Load the first slot to get duration + fps
    first_video_path = next(iter(slot_videos.values()))
    first_clip = VideoFileClip(first_video_path)
    duration = first_clip.duration
    fps = first_clip.fps

    # Background static image
    background = (
        ImageClip(static_png)
        .set_duration(duration)
        .set_fps(fps)
    )

    layers = [background]

    # Add each slot video
    for slot_name, video_path in slot_videos.items():
        geom = slot_geometries[slot_name]

        clip = (
            VideoFileClip(video_path)
            .resize((geom["width"], geom["height"]))
            .set_position((geom["x"], geom["y"]))
        )

        layers.append(clip)

    # Composite all layers
    final = CompositeVideoClip(layers)

    # Export
    final.write_videofile(
        output_path,
        codec="libx264",
        fps=fps,
        audio=False,
        preset="veryslow",
        bitrate="20M",
        ffmpeg_params=[
            "-crf", "8",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.2",
        ],
    )



class DynamicSlide:
    """
    A general-purpose slide that:
    - renders a static PNG via SlideRenderer
    - extracts slot geometries
    - renders dynamic content for each slot via SLOT_RENDERERS
    - composites static + dynamic
    - optionally upscales the final output
    """

    def __init__(
        self,
        *,
        layout_spec: dict,
        slot_payloads: dict,   # slot_name → payload dict
        output_dir: str,
        scale: int = 3,
        upscale: int = 2,
        slide_index: int = 1
    ):
        self.layout_spec = layout_spec
        self.slot_payloads = slot_payloads
        self.output_dir = Path(output_dir)
        self.scale = scale
        self.upscale = upscale
        self.slide_index = slide_index

        # Paths (filled in by _prepare_paths)
        self.fpath_image = None
        self.fpath_composite = None
        self.fpath_final = None
        self.slot_video_paths = {}
        self.slot_geometries = {}

    # ------------------------------------------------------------
    # Path preparation
    # ------------------------------------------------------------
    def _prepare_paths(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.slide_index}"

        self.fpath_image = self.output_dir / f"{prefix}_static.png"
        self.fpath_composite = self.output_dir / f"{prefix}_composite.mp4"
        self.fpath_final = self.output_dir / f"{prefix}_final.mp4"

    # ------------------------------------------------------------
    # Static slide rendering
    # ------------------------------------------------------------
    def _render_static(self):
        
        canvas_size = (
            self.layout_spec.get("canvas_width", 1080),
            self.layout_spec.get("canvas_height", 1920)
        )

        # inject scale into the spec so SlideRenderer sees it
        spec = {**self.layout_spec, "scale": self.scale}

        renderer = SlideRenderer(spec, canvas_size=canvas_size, scale = self.scale)
        img, slot_geometries = renderer.render_with_slots(
            output_path=str(self.fpath_image)
        )

        self.slot_geometries = slot_geometries


    # ------------------------------------------------------------
    # Render each slot’s dynamic content
    # ------------------------------------------------------------
    def _render_slots(self):
        
        for slot_name, payload in self.slot_payloads.items():
            slot_type = payload["type"]
            slot_renderer = SLOT_RENDERERS[slot_type]

            geom = self.slot_geometries[slot_name]
            video_name = f"{self.slide_index}_{slot_name}.mp4"
            
            video_path = self.output_dir / f"{video_name}"
            
            slot_renderer(
                geom=geom,
                payload=payload,
                output_path=str(video_name)
            )

            self.slot_video_paths[slot_name] = str(video_path)

    # ------------------------------------------------------------
    # Composite static + dynamic
    # ------------------------------------------------------------
    def _composite(self):
        
        composite_slide_with_slots(
            static_png=str(self.fpath_image),
            slot_videos=self.slot_video_paths,
            slot_geometries=self.slot_geometries,
            output_path=str(self.fpath_composite)
        )

    # ------------------------------------------------------------
    # Upscale final output
    # ------------------------------------------------------------
    def _upscale(self):
        if self.upscale > 1:
            upscale_video(
                str(self.fpath_composite),
                str(self.fpath_final),
                scale_factor=self.upscale
            )
        else:
            self.fpath_final = self.fpath_composite

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def render(self):
        self._prepare_paths()
        self._render_static()
        self._render_slots()
        self._composite()
        self._upscale()
        return str(self.fpath_final)



class ScrollSlide:
    """
    A behavior class that uses SlideRenderer to create a tall slide
    and then animates it vertically over time.
    """

    def __init__(
        self,
        layout_spec,
        canvas_size,
        scale=1.0,
    ):
        self.layout_spec = layout_spec
        
        self.canvas_size = (
            int(canvas_size[0] * scale),
            int(canvas_size[1] * scale)
        )
        self.scale = scale
        
        canvas_height = measure_slide_height(
            layout_spec, 
            self.canvas_size[0], 
            scale=1.0
        )

        self.renderer = SlideRenderer(
            layout_spec=layout_spec,
            canvas_size=(self.canvas_size[0], canvas_height),
            scale=scale,
        )

    @staticmethod
    def build_transcript_blocks(
        segments,
        speaker_font="bold",
        speaker_size=60,
        speaker_color=[255, 255, 0],
        content_font="regular",
        content_size=60,
        content_color=[255, 255, 255],
        gap_after=20,
    ):
        blocks = []
        for seg in segments:
            blocks.append({
                "type": "rich_text",
                "spans": [
                    {
                        "text": seg["speaker"],
                        "font": speaker_font,
                        "size": speaker_size,
                        "color": speaker_color,
                    },
                    {
                        "text": seg["content"],
                        "font": content_font,
                        "size": content_size,
                        "color": content_color,
                    }
                ],
                "gap_after": gap_after,
            })
        return blocks

    # ------------------------------------------------------------
    # Render tall static slide (main content)
    # ------------------------------------------------------------
    def render_static(self):
        img = self.renderer.render()
        return img, img.size[1]

    # ------------------------------------------------------------
    # Build MoviePy scroll clip
    # ------------------------------------------------------------
    def to_clip(
        self,
        duration=None,
        audio=None,
        extra_silence=0.0,
        bg_color=(0, 0, 0),
    ):

        # ------------------------------------------------------------
        # 1. Render main tall slide
        # ------------------------------------------------------------
        main_img, main_height = self.render_static()

        # Trim empty bottom space
        main_img = trim_bottom_empty_space(
            main_img,
            bg_color=self.layout_spec["background_color"]
        )
        main_height = main_img.size[1]

        # ------------------------------------------------------------
        # 2. OPTIONAL INTRO IMAGE (full width)
        # ------------------------------------------------------------
        intro_spec = self.layout_spec.get("intro_image")

        if intro_spec is not None:

            intro_layout = {
                "background_color": self.layout_spec["background_color"],
                "padding": intro_spec.get("padding", self.layout_spec.get("padding", 40)),
                "fonts": self.layout_spec["fonts"],
                "blocks": [
                    {
                        "type": "svg",
                        "path": intro_spec["path"],
                        "gap_after": intro_spec.get("gap_after", 40),
                        "scale": intro_spec.get("scale", 0.25)
                    }
                ]
            }

            print(intro_layout)

            intro_height = measure_slide_height(
                intro_layout,
                self.canvas_size[0],
                scale=1.0
            )

            intro_renderer = SlideRenderer(
                layout_spec=intro_layout,
                canvas_size=(self.canvas_size[0], intro_height),
                scale=self.scale
            )

            intro_img = intro_renderer.render()

            # ------------------------------------------------------------
            # 3. Stack intro + main into one tall image
            # ------------------------------------------------------------
            combined_height = intro_img.size[1] + main_img.size[1]
            img = Image.new(
                "RGBA",
                (self.canvas_size[0], combined_height),
                self.layout_spec["background_color"]
            )

            img.paste(intro_img, (0, 0))
            img.paste(main_img, (0, intro_img.size[1]))

            text_height = combined_height

        else:
            # No intro image → revert to original behavior
            img = main_img
            text_height = main_height

        # ------------------------------------------------------------
        # 4. Save tall image for MoviePy (NOW includes intro!)
        # ------------------------------------------------------------
        temp_path = "_scrollslide_temp.png"
        img.save(temp_path)

        txt_clip = ImageClip(temp_path)

        # Determine duration
        if duration is None:
            duration = (
                audio.duration + extra_silence
                if audio is not None
                else 8.0
            )

        W, H = self.canvas_size

        # Scroll math
        # y_start depends on whether intro image exists
        if intro_spec is not None:
            y_start = 400
        else:
            y_start = 1500

        y_end = H - text_height - 450

        def scroll_position(t):
            
            progress = min(max(t / duration, 0.0), 1.0)
            y = y_start + progress * (y_end - y_start)
            return ("center", y)

        txt_clip = txt_clip.set_duration(duration).set_position(scroll_position)

        # Background
        background = ColorClip(
            size=self.canvas_size,
            color=self.layout_spec["background_color"],
            duration=duration
        )

        overlay_clips = [background, txt_clip]

        # ------------------------------------------------------------
        # 5. HEADER OVERLAY (unchanged)
        # ------------------------------------------------------------
        header_spec = self.layout_spec.get("header")

        if header_spec is not None:

            header_layout = {
                "background_color": header_spec.get("bg_color", [0,0,0]),
                "padding": header_spec.get("padding", 40),
                "fonts": self.layout_spec["fonts"],
                "blocks": [
                    {
                        "type": "text",
                        "text": header_spec["text"],
                        "font": header_spec.get("font", "regular"),
                        "size": header_spec.get("size", 40),
                        "color": header_spec.get("color", [255,255,255]),
                        "gap_after": 20
                    }
                ]
            }

            canvas_height = measure_slide_height(
                header_layout,
                W,
                scale=1.0
            )

            header_renderer = SlideRenderer(
                layout_spec=header_layout,
                canvas_size=(self.canvas_size[0], canvas_height),
                scale=self.scale
            )

            header_img = header_renderer.render()

            temp_header_path = "_scrollslide_header.png"
            header_img.save(temp_header_path)

            header_clip = (
                ImageClip(temp_header_path)
                .set_position(("center", "top"))
                .set_duration(duration)
            )

            overlay_clips.append(header_clip)

        scroll_clip = CompositeVideoClip(overlay_clips)

        if audio is not None:
            scroll_clip = scroll_clip.set_audio(audio)

        return scroll_clip


def measure_slide_height(layout_spec, width, scale=1.0):
    TEMP_HEIGHT = 100000

    renderer = SlideRenderer(
        layout_spec=layout_spec,
        canvas_size=(width, TEMP_HEIGHT),
        scale=scale
    )

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    g = scale
    measured, padding, frame, safe_height, safe_left, safe_right = \
        renderer.measure_blocks(draw, g)

    return renderer.compute_total_height(measured)

def trim_bottom_empty_space(img, bg_color, margin=550):
    """
    Trim bottom empty space but keep a safety margin so text isn't cut off.
    margin = number of extra pixels to keep below the last detected content row.
    """
    # Normalize bg_color to RGB
    if len(bg_color) == 4:
        bg_color = bg_color[:3]

    arr = np.array(img)
    h = arr.shape[0]

    # Find last non-background row
    mask = np.any(arr != bg_color, axis=2)
    rows_with_content = np.where(mask.any(axis=1))[0]

    if len(rows_with_content) == 0:
        return img  # nothing to trim

    last_content_row = rows_with_content[-1]

    # Add safety margin (but don't exceed image height)
    end_row = min(last_content_row + margin, h - 1)

    trimmed = arr[:end_row+1, :, :]
    return Image.fromarray(trimmed)
