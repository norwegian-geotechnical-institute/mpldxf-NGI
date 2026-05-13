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
    x,
    y,
    s,
    prop,
    angle,
    ismath,
    mtext,
    points_to_pixels,
    use_fm_layers,
    determine_text_layer,
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

    if s and len(s) > 0 and s[0] == "$":
        pattern = r"\\mathbf\{(.*?)\}"
        stripped_text = re.sub(pattern, r"\1", s)
        stripped_text = re.sub(r"[$]", "", stripped_text)
        stripped_text = re.sub(r"\\/", " ", stripped_text)
        text = modelspace.add_text(
            stripped_text,
            height=fontsize,
            rotation=angle,
            dxfattribs=dxfattribs,
        )
    else:
        text = modelspace.add_text(
            s,
            height=fontsize,
            rotation=angle,
            dxfattribs=dxfattribs,
        )

    # Matplotlib passes (x, y) in display coordinates; treat that as the
    # anchor point and map Matplotlib's alignment to ezdxf's alignment enum.
    if mtext is not None:
        if angle == 90.0:
            if getattr(mtext, "_rotation_mode", None) == "anchor":
                halign = _map_align(mtext.get_ha(), vert=False)
            else:
                halign = "RIGHT"
            valign = _map_align(mtext.get_va(), vert=True)
        else:
            halign = _map_align(mtext.get_ha(), vert=False)
            valign = _map_align(mtext.get_va(), vert=True)
    else:
        # For multi-line texts Matplotlib may pass mtext=None; fall back to the
        # default Text alignment (left/baseline).
        halign = "LEFT"
        valign = ""

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

    p1 = float(x), float(y)
    text.set_placement(p1, align=align)
    return p1
