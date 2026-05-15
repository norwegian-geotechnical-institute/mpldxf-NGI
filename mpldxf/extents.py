"""DXF extents tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry_utils import is_valid_coordinate


@dataclass
class Extents2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def from_size(cls, *, width: float, height: float) -> "Extents2D":
        return cls(0.0, 0.0, float(width), float(height))

    def track_point(self, x, y) -> None:
        if not is_valid_coordinate([x, y]):
            return
        xf = float(x)
        yf = float(y)
        self.min_x = min(self.min_x, xf)
        self.min_y = min(self.min_y, yf)
        self.max_x = max(self.max_x, xf)
        self.max_y = max(self.max_y, yf)

    def track_points(self, points) -> None:
        for x, y in points:
            self.track_point(x, y)

    def track_circle(self, center, radius: float) -> None:
        try:
            cx, cy = center
            r = float(radius)
        except Exception:
            return
        if r < 0:
            r = -r
        self.track_point(cx - r, cy - r)
        self.track_point(cx + r, cy + r)

    def as_extmin(self):
        return (float(self.min_x), float(self.min_y), 0.0)

    def as_extmax(self):
        return (float(self.max_x), float(self.max_y), 0.0)

    def center(self):
        width = float(self.max_x - self.min_x)
        height = float(self.max_y - self.min_y)
        return (float(self.min_x + width / 2.0), float(self.min_y + height / 2.0))

    def size(self):
        return (float(self.max_x - self.min_x), float(self.max_y - self.min_y))


def apply_extents_to_drawing(*, drawing, modelspace, extents: Extents2D) -> None:
    """
    Apply extents to DXF header and modelspace, and hint an initial view.

    This improves the default initial zoom/viewport in CAD applications.
    """
    extmin = extents.as_extmin()
    extmax = extents.as_extmax()
    drawing.header["$EXTMIN"] = extmin
    drawing.header["$EXTMAX"] = extmax

    try:
        modelspace.dxf.extmin = extmin
        modelspace.dxf.extmax = extmax
    except Exception:
        pass

    try:
        width, height = extents.size()
        if height <= 0:
            height = 1.0
        vport = drawing.set_modelspace_vport(
            height=height * 1.05,
            center=extents.center(),
        )
        if width > 0:
            try:
                vport.dxf.aspect_ratio = width / height
            except Exception:
                pass
    except Exception:
        pass

