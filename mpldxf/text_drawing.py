"""Helpers for exporting Matplotlib text to DXF."""

import re

from ezdxf.enums import TextEntityAlignment

from .color_utils import rgb_to_dxf


def _map_align(align, vert=False):
    """Translate a Matplotlib text alignment to the ezdxf alignment."""
    if align in ["right", "center", "left", "top", "bottom", "middle"]:
        align = align.upper()
    elif align == "baseline":
        align = ""
    elif align == "center_baseline":
        align = "MIDDLE"
    else:
        raise NotImplementedError
    if vert and align == "CENTER":
        align = "MIDDLE"
    return align


def draw_text_entity(
    modelspace,
    gc,
    s,
    prop,
    angle,
    mtext,
    points_to_pixels,
    use_fm_layers,
    determine_text_layer,
    x=None,
    y=None,
):
    """Draw a DXF text entity from a Matplotlib text request."""
    fontsize = points_to_pixels(prop.get_size_in_points()) / 2

    dxfattribs = {}
    if use_fm_layers:
        layer_name = determine_text_layer(s, fontsize)
        dxfattribs["layer"] = layer_name
        dxfattribs["color"] = 256
    else:
        dxfattribs["color"] = rgb_to_dxf(gc.get_rgb())

    s = s.replace("\u2212", "-")
    s = s.encode("ascii", "ignore").decode()
    if not s:
        return

    if s and len(s) > 0 and s[0] == "$":
        pattern = r"\\mathbf\{(.*?)\}"
        text_content = re.sub(pattern, r"\1", s)
        text_content = re.sub(r"[$]", "", text_content)
        text_content = re.sub(r"\\/", " ", text_content)
    else:
        text_content = s

    if not text_content:
        return

    # Matplotlib does not guarantee that it passes the underlying Text artist
    # object (``mtext``). When it is missing, fall back to the explicit x/y
    # coordinates from the renderer call.
    if mtext is None:
        if x is None or y is None:
            return
        text = modelspace.add_text(
            text_content,
            height=fontsize,
            rotation=angle,
            dxfattribs=dxfattribs,
        )
        text.set_placement((float(x), float(y)), align=TextEntityAlignment.LEFT)
        return

    text = modelspace.add_text(
        text_content,
        height=fontsize,
        rotation=angle,
        dxfattribs=dxfattribs,
    )

    if angle == 90.0:
        if mtext._rotation_mode == "anchor":
            halign = _map_align(mtext.get_ha(), vert=False)
        else:
            halign = "RIGHT"
        valign = _map_align(mtext.get_va(), vert=True)
    else:
        halign = _map_align(mtext.get_ha(), vert=False)
        valign = _map_align(mtext.get_va(), vert=True)

    if valign and valign != "":
        align = valign + "_" + halign
    else:
        align = halign

    if not align or align == "" or align == "_":
        align = "LEFT"

    alignment_map = {
        "TOP_LEFT": TextEntityAlignment.TOP_LEFT,
        "TOP_CENTER": TextEntityAlignment.TOP_CENTER,
        "TOP_RIGHT": TextEntityAlignment.TOP_RIGHT,
        "MIDDLE_LEFT": TextEntityAlignment.MIDDLE_LEFT,
        "MIDDLE_CENTER": TextEntityAlignment.MIDDLE_CENTER,
        "MIDDLE_RIGHT": TextEntityAlignment.MIDDLE_RIGHT,
        "BOTTOM_LEFT": TextEntityAlignment.BOTTOM_LEFT,
        "BOTTOM_CENTER": TextEntityAlignment.BOTTOM_CENTER,
        "BOTTOM_RIGHT": TextEntityAlignment.BOTTOM_RIGHT,
        "LEFT": TextEntityAlignment.LEFT,
        "CENTER": TextEntityAlignment.CENTER,
        "RIGHT": TextEntityAlignment.RIGHT,
    }

    align = alignment_map.get(align, TextEntityAlignment.BOTTOM_LEFT)

    pos = mtext.get_unitless_position()
    x, y = mtext.get_transform().transform(pos)
    p1 = x, y
    text.set_placement(p1, align=align)
