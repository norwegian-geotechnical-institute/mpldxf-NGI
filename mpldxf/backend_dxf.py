"""
A backend to export DXF using a custom DXF renderer.

This allows saving of DXF figures.

Use as a matplotlib external backend:

  import matplotlib
  matplotlib.use('module://mpldxf.backend_dxf')

or register:

  matplotlib.backend_bases.register_backend('dxf', FigureCanvasDxf)

Based on matplotlib.backends.backend_template.py.

Copyright (C) 2014 David M Kent

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from io import BytesIO, StringIO
import os
import sys
import math
import re

import matplotlib
from matplotlib.backend_bases import (
    RendererBase,
    FigureCanvasBase,
    GraphicsContextBase,
    FigureManagerBase,
)
from matplotlib.transforms import Affine2D
import matplotlib.transforms as transforms
import matplotlib.collections as mplc
import numpy as np
from shapely.geometry import LineString, Polygon
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math.clipping import Clipping, ClippingRect2d, ConvexClippingPolygon2d

from . import dxf_colors

# When packaged with py2exe ezdxf has issues finding its templates
# We tell it where to find them using this.
# Note we also need to make sure they get packaged by adding them to the
# configuration in setup.py
if hasattr(sys, "frozen"):
    ezdxf.options.template_dir = os.path.dirname(sys.executable)


def rgb_to_dxf(rgb_val):
    """Convert an RGB[A] colour to DXF colour index.

    ``rgb_val`` should be a tuple of values in range 0.0 - 1.0. Any
    alpha value is ignored.
    """
    if rgb_val is None:
        dxfcolor = dxf_colors.WHITE
    # change black to white
    elif np.allclose(np.array(rgb_val[:3]), np.zeros(3)):
        dxfcolor = dxf_colors.nearest_index([255, 255, 255])
    else:
        dxfcolor = dxf_colors.nearest_index([255.0 * val for val in rgb_val[:3]])
    return dxfcolor


class RendererDxf(RendererBase):
    """
    The renderer handles drawing/rendering operations.

    Renders the drawing using the ``ezdxf`` package.
    """

    def __init__(self, width, height, dpi, dxfversion):
        RendererBase.__init__(self)
        self.height = height
        self.width = width
        self.dpi = dpi
        self.dxfversion = dxfversion
        self._init_drawing()
        self._groupd = []

    def _init_drawing(self):
        """Create a drawing, set some global information and add
        the layers we need.
        """
        drawing = ezdxf.new(dxfversion=self.dxfversion)
        modelspace = drawing.modelspace()
        drawing.header["$EXTMIN"] = (0, 0, 0)
        drawing.header["$EXTMAX"] = (self.width, self.height, 0)
        self.drawing = drawing
        self.modelspace = modelspace

    def clear(self):
        """Reset the renderer."""
        super(RendererDxf, self).clear()
        self._init_drawing()

    def _get_polyline_attribs(self, gc):
        attribs = {}
        attribs["color"] = rgb_to_dxf(gc.get_rgb())
        return attribs

    def _clip_mpl(self, gc, vertices, obj):
        # clip the polygon if clip rectangle present
        bbox = gc.get_clip_rectangle()
        if bbox is not None:
            cliprect = [
                [bbox.x0, bbox.y0],
                [bbox.x1, bbox.y0],
                [bbox.x1, bbox.y1],
                [bbox.x0, bbox.y1],
            ]

            if obj == "patch":
                vertices = ClippingRect2d(cliprect[0], cliprect[2]).clip_polyline(
                    vertices
                )
            elif obj == "line2d":
                cliprect = Polygon(cliprect)
                line = LineString(vertices)
                try:
                    intersection = line.intersection(cliprect)
                except Exception:
                    # Fix plotted values with large gaps/jumps crashing the intersection function
                    gap_threshold = 50
                    smoothed_vertices = [vertices[0]]  # Start with the first point
                    for i in range(1, len(vertices)):
                        if (
                            abs(vertices[i][0] - vertices[i - 1][0]) <= gap_threshold
                            and abs(vertices[i][1] - vertices[i - 1][1])
                            <= gap_threshold
                        ):
                            smoothed_vertices.append(vertices[i])
                    line = LineString(smoothed_vertices)
                    intersection = line.intersection(cliprect)

                # Check if intersection is a multi-part geometry
                if intersection.is_empty:
                    vertices = []  # No intersection
                elif (
                    "Multi" in intersection.geom_type
                    or "GeometryCollection" in intersection.geom_type
                ):
                    # If intersection is a multi-part geometry, iterate
                    vertices = [list(geom.coords) for geom in intersection.geoms]
                else:
                    # If intersection is not a multi-part geometry, get intersection coordinates directly
                    vertices = list(intersection.coords)

        return vertices

    def _draw_mpl_lwpoly(self, gc, path, transform, obj):
        # TODO rework for BEZIER curves
        if obj == "patch":
            close = True
        elif obj == "line2d":
            close = False

        dxfattribs = self._get_polyline_attribs(gc)
        vertices = path.transformed(transform).vertices

        # clip the polygon if clip rectangle present

        if isinstance(vertices[0][0], float or np.float64):
            vertices = self._clip_mpl(gc, vertices, obj=obj)

        else:
            vertices = [self._clip_mpl(gc, points, obj=obj) for points in vertices]

        # if vertices.
        if len(vertices) == 0:
            entity = None

        else:
            if isinstance(vertices[0][0], float or np.float64):
                if vertices[0][0] != 0:
                    entity = self.modelspace.add_lwpolyline(
                        points=vertices, close=False, dxfattribs=dxfattribs
                    )  # set close to false because it broke some arrows
                else:
                    entity = None

            else:
                entity = [
                    self.modelspace.add_lwpolyline(
                        points=points, close=False, dxfattribs=dxfattribs
                    )
                    for points in vertices
                ]  # set close to false because it broke some arrows
            return entity

    def _draw_mpl_line2d(self, gc, path, transform):
        line = self._draw_mpl_lwpoly(gc, path, transform, obj="line2d")

    def _draw_mpl_patch(self, gc, path, transform, rgbFace=None):
        """Draw a matplotlib patch object"""

        poly = self._draw_mpl_lwpoly(gc, path, transform, obj="patch")
        if not poly:
            return
        # check to see if the patch is filled
        if rgbFace is not None:
            if type(poly) == list:
                for pol in poly:
                    hatch = self.modelspace.add_hatch(color=rgb_to_dxf(rgbFace))
                    hpath = hatch.paths.add_polyline_path(
                        # get path vertices from associated LWPOLYLINE entity
                        pol.get_points(format="xyb"),
                        # get closed state also from associated LWPOLYLINE entity
                        is_closed=pol.closed,
                    )

                    # Set association between boundary path and LWPOLYLINE
                    hatch.associate(hpath, [pol])
            else:
                hatch = self.modelspace.add_hatch(color=rgb_to_dxf(rgbFace))
                hpath = hatch.paths.add_polyline_path(
                    # get path vertices from associated LWPOLYLINE entity
                    poly.get_points(format="xyb"),
                    # get closed state also from associated LWPOLYLINE entity
                    is_closed=poly.closed,
                )

                # Set association between boundary path and LWPOLYLINE
                hatch.associate(hpath, [poly])

        self._draw_mpl_hatch(gc, path, transform, pline=poly)

    def _draw_mpl_hatch(self, gc, path, transform, pline):
        """Draw MPL hatch"""

        hatch = gc.get_hatch()
        if hatch is not None:
            # find extents and center of the original unclipped parent path
            ext = path.get_extents(transform=transform)
            dx = ext.x1 - ext.x0
            cx = 0.5 * (ext.x1 + ext.x0)
            dy = ext.y1 - ext.y0
            cy = 0.5 * (ext.y1 + ext.y0)

            # matplotlib uses a 1-inch square hatch, so find out how many rows
            # and columns will be needed to fill the parent path
            rows, cols = math.ceil(dy / self.dpi) - 1, math.ceil(dx / self.dpi) - 1

            # get color of the hatch
            rgb = gc.get_hatch_color()
            dxfcolor = rgb_to_dxf(rgb)

            # get hatch paths
            hpath = gc.get_hatch_path()

            # this is a tranform that produces a properly scaled hatch in the center
            # of the parent hatch
            _transform = (
                Affine2D().translate(-0.5, -0.5).scale(self.dpi).translate(cx, cy)
            )
            hpatht = hpath.transformed(_transform)

            # print("\tHatch Path:", hpatht)
            # now place the hatch to cover the parent path
            for irow in range(-rows, rows + 1):
                for icol in range(-cols, cols + 1):
                    # transformation from the center of the parent path
                    _trans = Affine2D().translate(icol * self.dpi, irow * self.dpi)
                    # transformed hatch
                    _hpath = hpatht.transformed(_trans)

                    # turn into list of vertices to make up polygon
                    _path = _hpath.to_polygons(closed_only=False)

                    for vertices in _path:
                        if pline is not None:
                            for (
                                pline_obj
                            ) in pline:  # Assuming pline is a list of objects
                                if len(vertices) == 2:
                                    clippoly = Polygon(
                                        pline_obj.vertices()
                                    )  # Access vertices of each object in the list
                                    line = LineString(vertices)
                                    clipped = line.intersection(clippoly).coords
                                else:
                                    clipped = ezdxf.math.clipping.ClippingRect2d(
                                        pline_obj.vertices(), vertices
                                    )
                        else:
                            clipped = []

                        # if there is something to plot
                        if len(clipped) > 0:
                            if len(vertices) == 2:
                                attrs = {"color": dxfcolor}
                                self.modelspace.add_lwpolyline(
                                    points=clipped, dxfattribs=attrs
                                )
                            else:
                                # A non-filled polygon or a line - use LWPOLYLINE entity
                                hatch = self.modelspace.add_hatch(color=dxfcolor)
                                line = hatch.paths.add_polyline_path(clipped)

    def draw_path_collection(
        self,
        gc,
        master_transform,
        paths,
        all_transforms,
        offsets,
        offsetTrans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
    ):
        for i, path in enumerate(paths):
            if all_transforms:
                combined_transform = master_transform + all_transforms[i]
            else:
                combined_transform = master_transform
            facecolor = facecolors[i] if i < len(facecolors) else None
            edgecolor = edgecolors[i] if i < len(edgecolors) else None
            # Draw each path as a filled patch
            self._draw_mpl_patch(gc, path, combined_transform, rgbFace=facecolor)

    def draw_path(self, gc, path, transform, rgbFace=None):
        # print('\nEntered ###DRAW_PATH###')
        # print('\t', self._groupd)
        # print('\t', gc.__dict__, rgbFace)
        # print('\t', gc.get_sketch_params())
        # print('\tMain Path', path.__dict__)
        # hatch = gc.get_hatch()
        # if hatch is not None:
        #     print('\tHatch Path', gc.get_hatch_path().__dict__)

        if self._groupd[-1] == "patch":
            # print('Draw Patch')
            line = self._draw_mpl_patch(gc, path, transform, rgbFace)

        elif self._groupd[-1] == "line2d":
            line = self._draw_mpl_line2d(gc, path, transform)

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # print('\nEntered ###DRAW_MARKERS###')
        # print('\t', self._groupd)
        # print('\t', gc.__dict__)
        # print('\tMarker Path:', type(marker_path), marker_path.transformed(marker_trans).__dict__)
        # print('\tPath:', type(path), path.transformed(trans).__dict__)
        if (self._groupd[-1] == "line2d") & ("tick" in self._groupd[-2]):
            newpath = path.transformed(trans)
            dx, dy = newpath.vertices[0]
            _trans = marker_trans + Affine2D().translate(dx, dy)
            line = self._draw_mpl_line2d(gc, marker_path, _trans)
        # print('\tLeft ###DRAW_MARKERS###')

    def draw_image(self, gc, x, y, im):
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # print('\nEntered ###DRAW_TEXT###')
        # print('\t', self._groupd)
        # print('\t', gc.__dict__)
        if mtext is None:
            pass
        else:
            fontsize = self.points_to_pixels(prop.get_size_in_points()) / 2
            dxfcolor = rgb_to_dxf(gc.get_rgb())

            s = s.replace("\u2212", "-")
            s.encode("ascii", "ignore").decode()
            if s[0] == "$":
                pattern = r"\\mathbf\{(.*?)\}"
                stripped_text = re.sub(pattern, r"\1", s)
                stripped_text = re.sub(r"[$]", "", stripped_text)
                stripped_text = re.sub(r"\\/", " ", stripped_text)
                text = self.modelspace.add_text(
                    stripped_text,
                    height=fontsize,
                    rotation=angle,
                    dxfattribs={"color": dxfcolor},
                )
            else:
                text = self.modelspace.add_text(
                    s,
                    height=fontsize,
                    rotation=angle,
                    dxfattribs={"color": dxfcolor},
                )
            try:
                stripped_text
            except NameError:
                pass

            if angle == 90.0:
                if mtext._rotation_mode == "anchor":
                    halign = self._map_align(mtext.get_ha(), vert=False)
                else:
                    halign = "RIGHT"
                valign = self._map_align(mtext.get_va(), vert=True)
            else:
                halign = self._map_align(mtext.get_ha(), vert=False)
                valign = self._map_align(mtext.get_va(), vert=True)

            align = valign
            if align:
                align += "_"
            align += halign

            # need to create a TextEntityAlignment to work with ezdxf
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

            # need to get original points for text anchoring
            pos = mtext.get_unitless_position()
            x, y = mtext.get_transform().transform(pos)

            p1 = x, y
            text.set_placement(p1, align=align)
            # print('Left ###TEXT###')

    def _map_align(self, align, vert=False):
        """Translate a matplotlib text alignment to the ezdxf alignment."""
        if align in ["right", "center", "left", "top", "bottom", "middle"]:
            align = align.upper()
        elif align == "baseline":
            align = ""
        elif align == "center_baseline":
            align = "MIDDLE"
        else:
            # print(align)
            raise NotImplementedError
        if vert and align == "CENTER":
            align = "MIDDLE"
        return align

    def open_group(self, s, gid=None):
        # docstring inherited
        self._groupd.append(s)

    def close_group(self, s):
        self._groupd.pop(-1)

    def flipy(self):
        return False

    def get_canvas_width_height(self):
        return self.width, self.height

    def new_gc(self):
        return GraphicsContextBase()

    def points_to_pixels(self, points):
        return points / 72.0 * self.dpi

    def draw_quad_mesh(
        self,
        gc,
        master_transform,
        meshWidth,
        meshHeight,
        coordinates,
        offsets,
        offsetTrans,
        facecolors,
        antialiased,
        edgecolors,
    ):
        pass


class FigureCanvasDxf(FigureCanvasBase):
    """
    A canvas to use the renderer. This only implements enough of the
    API to allow the export of DXF to file.
    """

    #: The DXF version to use. This can be set to another version
    #: supported by ezdxf if desired.
    DXFVERSION = "AC1032"

    def get_dxf_renderer(self, cleared=False):
        """Get a renderer to use. Will create a new one if we don't
        alreadty have one or if the figure dimensions or resolution have
        changed.
        """
        l, b, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi
        try:
            self._lastKey, self.dxf_renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = self._lastKey != key

        if need_new_renderer:
            self.dxf_renderer = RendererDxf(w, h, self.figure.dpi, self.DXFVERSION)
            self._lastKey = key
        elif cleared:
            self.dxf_renderer.clear()
        return self.dxf_renderer

    def draw(self):
        """
        Draw the figure using the renderer
        """
        renderer = self.get_dxf_renderer()
        self.figure.draw(renderer)
        return renderer.drawing

    # Add DXF to the class-scope filetypes dictionary
    filetypes = FigureCanvasBase.filetypes.copy()
    filetypes["dxf"] = "DXF"

    def print_dxf(self, filename=None, *args, **kwargs):
        """
        Write out a DXF file.
        """
        drawing = self.draw()
        # Check if filename is a BytesIO instance
        if isinstance(filename, StringIO):
            # ezdxf can only write to a string or a file (not BytesIO directly)
            drawing.write(filename)
        else:
            drawing.saveas(filename)  # Use saveas() for file paths

    def get_default_filetype(self):
        return "dxf"


FigureManagerDXF = FigureManagerBase

########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################

FigureCanvas = FigureCanvasDxf
