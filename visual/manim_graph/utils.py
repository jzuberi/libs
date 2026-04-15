import math

def nice_number(value, round_to_nearest=True):
    """
    Returns a 'nice' number approximately equal to value.
    Rounds to 1, 2, or 5 × 10^n.
    """
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)

    if round_to_nearest:
        if fraction < 1.5:
            nice = 1
        elif fraction < 3:
            nice = 2
        elif fraction < 7:
            nice = 5
        else:
            nice = 10
    else:
        if fraction <= 1:
            nice = 1
        elif fraction <= 2:
            nice = 2
        elif fraction <= 5:
            nice = 5
        else:
            nice = 10

    return nice * (10 ** exponent)


def compute_axis_scale(values, params):
    axis_params = params.get("y_axis", {})
    tick_count = axis_params.get("tick_count", 5)
    nice_rounding = axis_params.get("nice_rounding", True)

    # Always start at zero unless user overrides
    if axis_params.get("min") is not None:
        vmin = axis_params["min"]
    else:
        vmin = 0

    vmax = max(values)
    if axis_params.get("max") is not None:
        vmax = axis_params["max"]

    raw_range = vmax - vmin
    if raw_range == 0:
        raw_range = abs(vmax) if vmax != 0 else 1

    step = nice_number(raw_range / tick_count, round_to_nearest=nice_rounding)

    y_min = vmin
    y_max = math.ceil(vmax / step) * step

    return y_min, y_max, step
