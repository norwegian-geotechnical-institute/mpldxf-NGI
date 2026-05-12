"""FM-specific layer definitions and routing rules.

This module collects the Field Manager layer names, colors, and the heuristics
used to map Matplotlib rendering context onto those layers.

The routing logic is intentionally simple and context-driven:
- group IDs can force a specific FM layer
- active Matplotlib group names decide the default layer for elements
- text is routed by axis context, title/label role, and simple numeric matching

Keeping these rules here makes the backend renderer easier to read, because the
DXF drawing code can delegate the FM-specific decisions to one place.
"""

import re


FM_LAYER_STYLES = {
    "FM-Frame": {"color": 3, "linetype": "CONTINUOUS"},   # Green - frames, ticks, gridlines
    "FM-Graph": {"color": 4, "linetype": "CONTINUOUS"},   # Cyan - data graphs/lines
    "FM-Location": {"color": 6, "linetype": "CONTINUOUS"}, # Magenta - location name text
    "FM-Method": {"color": 5, "linetype": "CONTINUOUS"},   # Blue - method icons and names
    "FM-Depth": {"color": 1, "linetype": "CONTINUOUS"},   # Red - Y-axis values (depth/elevation)
    "FM-Value": {"color": 8, "linetype": "CONTINUOUS"},  # Grey - X-axis values
    "FM-Text": {"color": 2, "linetype": "CONTINUOUS"},    # Yellow - axis labels and other text
    "FM-Grid-Vertical": {"color": 7, "linetype": "DASHED2"}, # Light Blue - vertical grid lines
    "FM-Grid-Horizontal": {"color": 7, "linetype": "DASHED2"},  # Light Blue - horizontal grid lines
}

# Backwards compatible alias used by older callers/tests.
FM_LAYERS = {name: style["color"] for name, style in FM_LAYER_STYLES.items()}


def create_fm_layers(drawing):
    """Create FM-specific layers with their configured colors."""
    for layer_name, style in FM_LAYER_STYLES.items():
        layer = drawing.layers.add(layer_name)
        layer.dxf.color = style["color"]
        layer.dxf.linetype = style.get("linetype", "CONTINUOUS")


def determine_element_layer(group_stack, group_gids):
    """Determine which FM layer should be used for the current element."""
    for group_name in group_stack:
        if group_gids.get(group_name) == "FM-Method":
            return "FM-Method"
        if group_gids.get(group_name) == "FM-Grid-Vertical":
            return "FM-Grid-Vertical"
        if group_gids.get(group_name) == "FM-Grid-Horizontal":
            return "FM-Grid-Horizontal"
        if group_gids.get(group_name) == "FM-Frame":
            return "FM-Frame"

    if not group_stack:
        return "0"

    current_element = group_stack[-1].lower()

    if current_element == "patch":
        return "PENDING"
    if current_element == "line2d":
        return "FM-Graph"
    if current_element == "collection":
        return "FM-Graph"
    if current_element == "method_collection":
        return "FM-Graph"
    if current_element == "text":
        return "FM-Text"
    return "0"


def determine_text_layer(group_stack, group_gids, text_content, fontsize):
    """Determine which FM layer should be used for text."""
    for group_name in group_stack:
        if group_gids.get(group_name) == "FM-Method":
            return "FM-Method"

    context_str = " ".join(group_stack).lower() if group_stack else ""

    if any(keyword in context_str for keyword in ["yaxis", "ytick"]):
        return "FM-Depth"
    if any(keyword in context_str for keyword in ["xaxis", "xtick"]):
        return "FM-Value"
    if "legend" in context_str:
        return "FM-Text"

    if "title" in context_str:
        if fontsize > 8:
            return "FM-Location"
        return "FM-Method"

    if "axes" in context_str and len(group_stack) == 3:
        if fontsize > 8:
            return "FM-Location"
        return "FM-Text"

    if any(keyword in context_str for keyword in ["xlabel", "ylabel"]):
        return "FM-Text"

    if re.match(r"^\s*[-+]?\d*\.?\d+\s*$", text_content):
        if "y" in context_str or "ytick" in context_str:
            return "FM-Depth"
        if "x" in context_str or "xtick" in context_str:
            return "FM-Value"

    return "FM-Text"
