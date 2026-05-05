"""Color conversion helpers for DXF export."""

import numpy as np

from . import dxf_colors


def rgb_to_dxf(rgb_val):
    """Convert an RGB[A] colour to a DXF colour index."""
    if rgb_val is None:
        dxfcolor = dxf_colors.WHITE
    elif np.allclose(np.array(rgb_val[:3]), np.zeros(3)):
        # Map black to white to avoid background-dependent DXF index 7 behaviour.
        dxfcolor = dxf_colors.nearest_index([255, 255, 255])
    else:
        dxfcolor = dxf_colors.nearest_index([255.0 * val for val in rgb_val[:3]])
    return dxfcolor
