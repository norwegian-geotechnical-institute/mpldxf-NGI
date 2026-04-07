"""Geometry and coordinate validation helpers for DXF export."""

import numpy as np


def is_valid_coordinate(coord):
    """Check if a coordinate contains only finite numbers."""
    coord_array = np.asarray(coord)
    return np.all(np.isfinite(coord_array))


def filter_invalid_coordinates(vertices):
    """Filter out vertices with NaN or Inf values."""
    if len(vertices) == 0:
        return vertices

    vertices_array = np.asarray(vertices)
    if vertices_array.ndim == 1:
        return vertices if is_valid_coordinate(vertices) else []
    return [vertex for vertex in vertices if is_valid_coordinate(vertex)]
