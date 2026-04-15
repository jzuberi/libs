
import math
import copy


# ============================================================
#  SVG PRIMITIVES
# ============================================================

def rgb_to_svg(rgb):
    r, g, b = rgb
    return f"rgb({r},{g},{b})"

# ============================================================
#  COLOR HELPERS
# ============================================================

def color_to_svg(c):
    """
    Accepts (r,g,b) or (r,g,b,a) tuples.
    Returns 'rgb(r,g,b)' or 'rgba(r,g,b,a)'.
    """
    if isinstance(c, str):
        return c  # allow raw strings if ever needed

    if len(c) == 3:
        r, g, b = c
        return f"rgb({r},{g},{b})"

    if len(c) == 4:
        r, g, b, a = c
        return f"rgba({r},{g},{b},{a/255:.3f})"

    raise ValueError("Color must be a 3- or 4-tuple.")


# ============================================================
#  SVG PRIMITIVES
# ============================================================

def svg_header(width, height, bg=None):
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )
    if bg is not None:
        header += (
            f'<rect x="0" y="0" width="{width}" height="{height}" '
            f'fill="{color_to_svg(bg)}" />\n'
        )
    return header


def svg_footer():
    return "</svg>\n"


def text(x, y, content, anchor="start", size=16, fill=(0,0,0), opacity=1.0, weight="bold"):
    return (
        f'<text x="{x}" y="{y}" '
        f'text-anchor="{anchor}" '
        f'font-family="{layout["FONT_FAMILY"]}" '
        f'font-size="{size}" '
        f'font-weight="{weight}" '
        f'fill="{color_to_svg(fill)}" opacity="{opacity}">'
        f'{content}</text>\n'
    )


def circle(cx, cy, r, stroke, fill=(255,255,255), width=2):
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'stroke="{color_to_svg(stroke)}" stroke-width="{width}" '
        f'fill="{color_to_svg(fill)}" />\n'
    )


def square_halo(cx, cy, color, layout, scale=1.0):
    size = layout["SQUARE_SIZE"] * scale
    half = size / 2
    return (
        f'<rect x="{cx - half}" y="{cy - half}" '
        f'width="{size}" height="{size}" '
        f'rx="6" ry="6" stroke="{color_to_svg(color)}" '
        f'stroke-width="{layout["HALO_STROKE_WIDTH"]}" fill="none" />\n'
        + circle(cx, cy, layout["STOP_RADIUS"] * scale,
                 stroke=color, fill=color, width=2)
    )


def dot_with_halo(cx, cy, color, layout, scale=1.0):
    return (
        circle(
            cx, cy,
            layout["HALO_RADIUS"] * scale,
            stroke=color,
            fill="none",
            width=layout["HALO_STROKE_WIDTH"]
        )
        + circle(
            cx, cy,
            layout["INNER_DOT_RADIUS"] * scale,
            stroke=color,
            fill=color,
            width=layout["STROKE_WIDTH"]
        )
    )


def hollow_circle_stop(cx, cy, color, layout, scale=1.0):
    return circle(
        cx, cy,
        layout["HALO_RADIUS"] * scale,
        stroke=color,
        fill="none",
        width=layout["REBUTTAL_STROKE_WIDTH"]
    )


# ============================================================
#  FIXED CIRCULAR LAYOUT HELPERS
# ============================================================

def flatten_nodes(config):
    """
    Produces:
      nodes = [(kind, color, clip_dict_or_None), ...]
      segment_ranges = [(start_idx, end_idx), ...]
    """
    nodes = []
    segment_ranges = []
    idx = 0

    for seg in config["segments"]:
        color = seg["color"]
        clips = seg["clips"]

        start = idx

        # Opening square node uses the FIRST clip object
        nodes.append(("square", color, clips[0]))
        idx += 1

        # Remaining clips (dots), skipping rebuttal
        for clip in clips[1:]:
            title = clip["title"]
            if title.startswith("Rebuttal - "):
                continue
            nodes.append(("dot", color, clip))
            idx += 1

        end = idx - 1
        segment_ranges.append((start, end))

    # Add rebuttal node if present in segment 0
    last_clip = config["segments"][0]["clips"][-1]
    if last_clip["title"].startswith("Rebuttal - "):
        petitioner_color = config["segments"][0]["color"]
        nodes.append(("rebuttal", petitioner_color, last_clip))

    return nodes, segment_ranges


def compute_circle_geometry(layout):
    W = layout["SVG_WIDTH"]
    cx = W / 2
    cy = W / 2
    R = cx - layout["CIRCLE_MARGIN"]
    return cx, cy, R


def compute_positions(N, cx, cy, R):
    positions = []
    angles = []

    for i in range(N):
        theta = (2 * math.pi * i / N) - math.pi/2  # start at 12 o'clock
        x = cx + R * math.cos(theta)
        y = cy + R * math.sin(theta)
        positions.append((x, y))
        angles.append(theta)

    return positions, angles


def draw_segment_arc(cx, cy, R, theta_start, theta_end, color, layout):
    x1 = cx + R * math.cos(theta_start)
    y1 = cy + R * math.sin(theta_start)
    x2 = cx + R * math.cos(theta_end)
    y2 = cy + R * math.sin(theta_end)

    large_arc = 1 if (theta_end - theta_start) % (2*math.pi) > math.pi else 0

    d = (
        f"M {x1} {y1} "
        f"A {R} {R} 0 {large_arc} 1 {x2} {y2}"
    )

    return (
        f'<path d="{d}" stroke="{color_to_svg(color)}" '
        f'stroke-width="{layout["STROKE_WIDTH"]}" '
        f'fill="none" stroke-linecap="round" />\n'
    )


def draw_node(kind, x, y, color, layout, clip):
    scale = layout["HIGHLIGHT_MULTIPLIER"] if (clip and clip.get("highlight")) else 1.0

    if kind == "square":
        return square_halo(x, y, color, layout, scale)
    if kind == "dot":
        return dot_with_halo(x, y, color, layout, scale)
    if kind == "rebuttal":
        return hollow_circle_stop(x, y, color, layout, scale)
    return ""


# ============================================================
#  generate_svg WITH PER-PAIR ARCS + LOCAL HIGHLIGHT ADJUSTMENT
# ============================================================

def generate_svg(config, layout):
    nodes, segment_ranges = flatten_nodes(config)
    N = len(nodes)

    cx, cy, R = compute_circle_geometry(layout)
    positions, angles = compute_positions(N, cx, cy, R)

    svg = svg_header(layout["SVG_WIDTH"], layout["SVG_WIDTH"], bg=None)

    base_gap = math.radians(layout["ARC_GAP_DEGREES"])

    # Extra angular padding only for highlighted nodes (local to adjacent arcs)
    extra_pad = []
    for (kind, color, clip) in nodes:
        if clip and clip.get("highlight"):
            scale = layout["HIGHLIGHT_MULTIPLIER"]
            extra_r = layout["HALO_RADIUS"] * (scale - 1.0)
            if extra_r <= 0:
                extra_pad.append(0.0)
            else:
                extra_pad.append(math.asin(min(extra_r / R, 0.9999)))
        else:
            extra_pad.append(0.0)

    # Draw arcs
    for seg_index, (start_i, end_i) in enumerate(segment_ranges):
        color = config["segments"][seg_index]["color"]

        if end_i <= start_i:
            continue

        for i in range(start_i, end_i):
            theta1 = angles[i]
            theta2 = angles[i + 1]

            # Base padding plus any extra from highlighted endpoints
            theta_start = theta1 + base_gap + extra_pad[i]
            theta_end   = theta2 - base_gap - extra_pad[i + 1]

            if (theta_end - theta_start) > 0:
                svg += draw_segment_arc(cx, cy, R, theta_start, theta_end, color, layout)

    # Draw nodes
    for (kind, color, clip), (x, y) in zip(nodes, positions):
        svg += draw_node(kind, x, y, color, layout, clip)

    svg += svg_footer()
    return svg




import math

# ============================================================
#  HORIZONTAL LAYOUT GEOMETRY
# ============================================================

def compute_horizontal_positions(N, layout):
    """
    Returns:
      positions = [(x, y), ...]
      xs = [x0, x1, ...]
    """
    W = layout["SVG_WIDTH"]
    H = layout["SVG_HEIGHT"]
    margin = layout["HORIZONTAL_MARGIN"]

    usable = W - 2 * margin
    step = usable / (N - 1) if N > 1 else 0

    y = H / 2
    positions = []
    xs = []

    for i in range(N):
        x = margin + i * step
        positions.append((x, y))
        xs.append(x)

    return positions, xs


def draw_horizontal_arc(x1, x2, y, color, layout):
    """
    Draws a straight horizontal segment between x1 and x2.
    """
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
        f'stroke="{color_to_svg(color)}" '
        f'stroke-width="{layout["STROKE_WIDTH"]}" '
        f'stroke-linecap="round" />\n'
    )


# ============================================================
#  generate_svg_horizontal
# ============================================================

def generate_svg_horizontal(config, layout):
    nodes, segment_ranges = flatten_nodes(config)
    N = len(nodes)

    # Compute base positions
    positions, xs = compute_horizontal_positions(N, layout)
    y = positions[0][1]

    # Horizontal margin already provided in layout
    margin = layout.get("HORIZONTAL_MARGIN", 0)

    # Shift all x positions inward by the margin
    xs = [x + margin for x in xs]
    positions = [(x + margin, y) for (x, y) in positions]

    # Expand SVG width to include both margins
    svg_width = layout["SVG_WIDTH"] + 2 * margin
    svg_height = layout["SVG_HEIGHT"]

    svg = svg_header(svg_width, svg_height, bg=None)

    base_gap = layout["ARC_GAP_PIXELS"]

    # Extra padding for highlighted nodes (same logic as circular)
    extra_pad = []
    for (kind, color, clip) in nodes:
        if clip and clip.get("highlight"):
            scale = layout["HIGHLIGHT_MULTIPLIER"]
            extra_r = layout["HALO_RADIUS"] * (scale - 1.0)
            extra_pad.append(extra_r)
        else:
            extra_pad.append(0.0)

    # Draw arcs
    for seg_index, (start_i, end_i) in enumerate(segment_ranges):
        color = config["segments"][seg_index]["color"]

        if end_i <= start_i:
            continue

        for i in range(start_i, end_i):
            x1 = xs[i]
            x2 = xs[i + 1]

            # Apply padding
            start_x = x1 + base_gap + extra_pad[i]
            end_x   = x2 - base_gap - extra_pad[i + 1]

            if end_x > start_x:
                svg += draw_horizontal_arc(start_x, end_x, y, color, layout)

    # Draw nodes
    for (kind, color, clip), (x, y) in zip(nodes, positions):
        svg += draw_node(kind, x, y, color, layout, clip)

    svg += svg_footer()
    return svg
