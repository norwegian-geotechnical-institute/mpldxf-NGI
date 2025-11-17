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
from shapely import Point
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
    """Convert an RGB[A] colour to DXF colour index."""
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
    Renders the drawing using the ``ezdxf`` package with NGI layer support.
    """

    def __init__(self, width, height, dpi, dxfversion, use_ngi_layers=False):
        RendererBase.__init__(self)
        self.height = height
        self.width = width
        self.dpi = dpi
        self.dxfversion = dxfversion
        self.use_ngi_layers = use_ngi_layers
        self._init_drawing()
        self._groupd = []
        self._method_context = False
        self._axes_patch_count = {}  # Track patches per axes
        self._current_axes_id = None
        self._axes_counter = 0  # Global axes counter
        self._pending_patch_analysis = None  # For position-based analysis

    def _init_drawing(self):
        """Create a drawing, set some global information and add the layers we need."""
        drawing = ezdxf.new(dxfversion=self.dxfversion)
        modelspace = drawing.modelspace()
        drawing.header["$EXTMIN"] = (0, 0, 0)
        drawing.header["$EXTMAX"] = (self.width, self.height, 0)

        if self.use_ngi_layers:
            self._create_ngi_layers(drawing)

        self.drawing = drawing
        self.modelspace = modelspace

    def _create_ngi_layers(self, drawing):
        """Create NGI-specific layers with specific colors"""
        ngi_layers = {
            "FM-Frame": 7,  # White/Black - frames, ticks, gridlines
            "FM-Graph": 1,  # Red - data graphs/lines
            "FM-Location": 2,  # Yellow - location name text
            "FM-Method": 3,  # Green - method icons and names
            "FM-Depth": 4,  # Cyan - Y-axis values (depth/elevation)
            "FM-Value": 5,  # Blue - X-axis values
            "FM-Text": 6,  # Magenta - axis labels and other text
        }

        for layer_name, color in ngi_layers.items():
            layer = drawing.layers.add(layer_name)
            layer.dxf.color = color

    def clear(self):
        """Reset the renderer."""
        super(RendererDxf, self).clear()
        self._init_drawing()

    def _determine_element_layer(self):
        """Determine which layer to use based on matplotlib element context"""
        if not self.use_ngi_layers:
            return "0"

        if not self._groupd:
            return "0"

        context_str = " ".join(self._groupd).lower()
        current_element = self._groupd[-1].lower()

        print(f"DETERMINE LAYER: Element='{current_element}', Context='{context_str}'")

        # Patches - defer to size analysis
        if current_element == "patch":
            return "PENDING"  # Will be resolved in _draw_mpl_patch

        # Line2D elements - distinguish data vs frame
        elif current_element == "line2d":
            # Frame elements: ticks and axis lines
            if any(keyword in context_str for keyword in ["tick", "matplotlib.axis"]):
                print(f"  -> FM-Frame (tick/axis line)")
                return "FM-Frame"
            # Data lines in axes context
            elif "axes" in context_str:
                print(f"  -> FM-Graph (data line)")
                return "FM-Graph"
            else:
                print(f"  -> FM-Graph (other line)")
                return "FM-Graph"

        # Collections - these are often method symbols
        elif current_element == "collection":
            print(f"  -> FM-Method (collection)")
            return "FM-Method"

        # Text elements
        elif current_element == "text":
            return "FM-Text"  # Will be overridden in text layer logic

        # Specific frame elements
        elif any(keyword in context_str for keyword in ["tick", "matplotlib.axis"]):
            print(f"  -> FM-Frame (frame element)")
            return "FM-Frame"

        print(f"  -> 0 (default)")
        return "0"

    def _analyze_patch_size(self, vertices):
        """Simple shape-based classification of patches"""
        if vertices is None or len(vertices) == 0:
            return "FM-Frame"

        # Convert to numpy array
        verts = np.array(vertices)

        # Calculate bounding box
        min_x, min_y = np.min(verts, axis=0)
        max_x, max_y = np.max(verts, axis=0)

        # Calculate dimensions
        width = max_x - min_x
        height = max_y - min_y

        print(f"    -> Patch size: {width:.1f}x{height:.1f}")

        # Avoid division by zero
        if height == 0 or width == 0:
            print(f"    -> FM-Frame (zero dimension)")
            return "FM-Frame"

        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height)

        print(f"    -> Aspect ratio: {aspect_ratio:.2f}")

        # Shape-based classification:
        # - Very thin/long elements (spines, gridlines) -> Frame
        # - Square-ish elements (method icons) -> Method
        # - Large backgrounds -> Frame

        if aspect_ratio > 10:  # Very long/thin = spines, gridlines, borders
            print(f"    -> FM-Frame (thin line/border)")
            return "FM-Frame"
        elif (
            aspect_ratio < 3 and width < 100 and height < 100
        ):  # Roughly square and small = method icon
            print(f"    -> FM-Method (square small patch - likely icon)")
            return "FM-Method"
        else:  # Everything else = frame elements
            print(f"    -> FM-Frame (frame element)")
            return "FM-Frame"

    def open_group(self, s, gid=None):
        """Open a grouping element with label *s*."""
        self._groupd.append(s)
        print(f"OPEN GROUP: {s}, Full context: {self._groupd}")

        # Track axes changes with a unique counter
        if s == "axes":
            self._axes_counter += 1
            self._current_axes_id = self._axes_counter
            if self._current_axes_id not in self._axes_patch_count:
                self._axes_patch_count[self._current_axes_id] = 0
            print(f"  -> New axes #{self._current_axes_id}")

        # Check if we're entering a method context
        if s.lower() in ["offsetbox", "drawingarea"]:
            self._method_context = True
            print(f"  -> METHOD CONTEXT ACTIVATED")

    def close_group(self, s):
        """Close a grouping element with label *s*."""
        print(f"CLOSE GROUP: {s}, Context before: {self._groupd}")
        if self._groupd and self._groupd[-1] == s:
            self._groupd.pop()

        # Check if we're exiting a method context
        if s.lower() in ["offsetbox", "drawingarea"]:  # Remove "anchored"
            self._method_context = False
            print(f"  -> METHOD CONTEXT DEACTIVATED")

    def _determine_text_layer(self, text_content, fontsize):
        """Determine text layer based on matplotlib context and content"""
        if not self.use_ngi_layers:
            return "0"

        context_str = " ".join(self._groupd).lower() if self._groupd else ""

        print(f"DETERMINE TEXT LAYER: Text='{text_content}', Context='{context_str}'")

        # Y-axis elements -> Depth
        if any(keyword in context_str for keyword in ["yaxis", "ytick"]):
            print(f"  -> FM-Depth (y-axis)")
            return "FM-Depth"

        # X-axis elements -> Value
        if any(keyword in context_str for keyword in ["xaxis", "xtick"]):
            print(f"  -> FM-Value (x-axis)")
            return "FM-Value"

        # Title elements and large text -> Location
        if "title" in context_str:
            if fontsize > 8:
                print(f"  -> FM-Location (title)")
                return "FM-Location"
            else:
                print(f"  -> FM-Method (title small)")
                return "FM-Method"

        # Text in general axes context - check position and content
        if "axes" in context_str:
            # Check if it's a large title-like text at top of plot
            # This is often location text even if it's just a number
            if len(self._groupd) == 3:  # ['figure', 'axes', 'text']
                if fontsize > 8:
                    print(f"  -> FM-Location (axes title text)")
                    return "FM-Location"
                else:
                    print(f"  -> FM-Method (axes title text small)")
                    return "FM-Method"

        # Legend elements -> Method
        if "legend" in context_str:
            print(f"  -> FM-Method (legend)")
            return "FM-Method"

        # Axis labels -> Text
        if any(keyword in context_str for keyword in ["xlabel", "ylabel"]):
            print(f"  -> FM-Text (axis labels)")
            return "FM-Text"

        # Content-based classification
        text_lower = text_content.lower()

        # Location text patterns
        if any(
            keyword in text_lower
            for keyword in ["boring", "bh-", "hole", "site", "location"]
        ):
            if fontsize > 8:
                print(f"  -> FM-Location (location pattern)")
                return "FM-Location"
            else:
                print(f"  -> FM-Method (location pattern small)")
                return "FM-Method"

        # Method text patterns
        if any(
            keyword in text_lower
            for keyword in ["cpt", "spt", "pmt", "dmt", "method", "test"]
        ):
            print(f"  -> FM-Method (method pattern)")
            return "FM-Method"

        # Numeric patterns
        import re

        if re.match(r"^\s*[-+]?\d*\.?\d+\s*$", text_content):
            if "y" in context_str or "ytick" in context_str:
                print(f"  -> FM-Depth (numeric y)")
                return "FM-Depth"
            elif "x" in context_str or "xtick" in context_str:
                print(f"  -> FM-Value (numeric x)")
                return "FM-Value"

        print(f"  -> FM-Text (default)")
        return "FM-Text"

    def _get_polyline_attribs(self, gc):
        """Get polyline attributes with correct layer and color"""
        attribs = {}
        if self.use_ngi_layers:
            layer_name = self._determine_element_layer()
            attribs["layer"] = layer_name
            attribs["color"] = 256  # ByLayer color
        else:
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
                # strip nans
                vertices = [v for v in vertices if not np.isnan(v).any()]

                cliprect = Polygon(cliprect)
                if (
                    len(vertices) == 1
                ):  # if there is only one data point for the line, create a Point object
                    line = Point(vertices[0])
                else:
                    line = LineString(vertices)
                try:
                    intersection = line.intersection(cliprect)
                except:
                    intersection = Polygon()

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

    def _draw_mpl_lwpoly(self, gc, path, transform, obj, dxfattribs=None):
        if dxfattribs is None:
            dxfattribs = self._get_polyline_attribs(gc)

        vertices = path.transformed(transform).vertices

        if len(vertices) > 0:
            if isinstance(vertices[0][0], (float, np.float64)):
                vertices = self._clip_mpl(gc, vertices, obj=obj)
            else:
                vertices = [self._clip_mpl(gc, points, obj=obj) for points in vertices]

            if len(vertices) == 0:
                entity = None
            else:
                if isinstance(vertices[0][0], (float, np.float64)):
                    if vertices[0][0] != 0:
                        entity = self.modelspace.add_lwpolyline(
                            points=vertices, close=False, dxfattribs=dxfattribs
                        )
                    else:
                        entity = None
                else:
                    entity = [
                        self.modelspace.add_lwpolyline(
                            points=points, close=False, dxfattribs=dxfattribs
                        )
                        for points in vertices
                    ]
            return entity

    def _draw_mpl_line2d(self, gc, path, transform):
        line = self._draw_mpl_lwpoly(gc, path, transform, obj="line2d")

    def _draw_mpl_patch(self, gc, path, transform, rgbFace=None):
        """Draw a matplotlib patch object"""

        # Get vertices for size analysis
        vertices = path.transformed(transform).vertices

        # Determine layer
        layer_name = self._determine_element_layer()

        if layer_name == "PENDING":
            # Use simple size-based analysis
            layer_name = self._analyze_patch_size(vertices)
            print(f"    -> Final layer: {layer_name}")

        # Set up DXF attributes
        dxfattribs = {}
        if self.use_ngi_layers:
            dxfattribs["layer"] = layer_name
            dxfattribs["color"] = 256  # ByLayer color
        else:
            dxfattribs["color"] = rgb_to_dxf(gc.get_rgb())

        # Draw the polygon outline
        poly = self._draw_mpl_lwpoly(
            gc, path, transform, obj="patch", dxfattribs=dxfattribs
        )
        if not poly:
            return

        # check to see if the patch is filled
        if rgbFace is not None:
            if type(poly) == list:
                for pol in poly:
                    hatch = self.modelspace.add_hatch(
                        color=rgb_to_dxf(rgbFace), dxfattribs=dxfattribs
                    )
                    hpath = hatch.paths.add_polyline_path(
                        pol.get_points(format="xyb"),
                        is_closed=pol.closed,
                    )
                    hatch.associate(hpath, [pol])
            else:
                hatch = self.modelspace.add_hatch(
                    color=rgb_to_dxf(rgbFace), dxfattribs=dxfattribs
                )
                hpath = hatch.paths.add_polyline_path(
                    poly.get_points(format="xyb"),
                    is_closed=poly.closed,
                )
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
        offset_trans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
    ):
        """Path collections might be method icons - force to method layer"""
        print(f"DRAW PATH COLLECTION: Context: {self._groupd}")

        # Force path collections to method layer (these are often scatter plots/symbols)
        original_groupd = self._groupd.copy()
        self._groupd.append("method_collection")  # Add marker

        for path in paths:
            combined_transform = master_transform
            if facecolors.size:
                rgbFace = facecolors[0] if facecolors is not None else None
            else:
                rgbFace = None
            self._draw_mpl_patch(gc, path, combined_transform, rgbFace=rgbFace)

        # Restore original context
        self._groupd = original_groupd

    def draw_path(self, gc, path, transform, rgbFace=None):
        """Draw a Path instance using the given affine transform."""
        if len(self._groupd) > 0:
            if self._groupd[-1] == "patch":
                self._draw_mpl_patch(gc, path, transform, rgbFace)
            elif self._groupd[-1] == "line2d":
                self._draw_mpl_line2d(gc, path, transform)
        else:
            # Default to patch
            self._draw_mpl_patch(gc, path, transform, rgbFace)

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        """Draw markers at each of the vertices in path."""
        if (
            len(self._groupd) > 0
            and self._groupd[-1] == "line2d"
            and len(self._groupd) > 1
            and "tick" in self._groupd[-2]
        ):
            newpath = path.transformed(trans)
            dx, dy = newpath.vertices[0]
            _trans = marker_trans + Affine2D().translate(dx, dy)
            self._draw_mpl_line2d(gc, marker_path, _trans)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """Draw text with proper layer assignment"""
        if mtext is None:
            return

        fontsize = self.points_to_pixels(prop.get_size_in_points()) / 2

        dxfattribs = {}
        if self.use_ngi_layers:
            layer_name = self._determine_text_layer(s, fontsize)
            dxfattribs["layer"] = layer_name
            dxfattribs["color"] = 256  # ByLayer color
        else:
            dxfattribs["color"] = rgb_to_dxf(gc.get_rgb())

        # Process text content
        s = s.replace("\u2212", "-")
        s.encode("ascii", "ignore").decode()

        if s and len(s) > 0 and s[0] == "$":
            pattern = r"\\mathbf\{(.*?)\}"
            stripped_text = re.sub(pattern, r"\1", s)
            stripped_text = re.sub(r"[$]", "", stripped_text)
            stripped_text = re.sub(r"\\/", " ", stripped_text)
            text = self.modelspace.add_text(
                stripped_text,
                height=fontsize,
                rotation=angle,
                dxfattribs=dxfattribs,
            )
        else:
            text = self.modelspace.add_text(
                s,
                height=fontsize,
                rotation=angle,
                dxfattribs=dxfattribs,
            )

        # Text alignment
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

    def _map_align(self, align, vert=False):
        """Translate a matplotlib text alignment to the ezdxf alignment."""
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

    def flipy(self):
        return False

    def get_canvas_width_height(self):
        return self.width, self.height

    def new_gc(self):
        return GraphicsContextBase()

    def points_to_pixels(self, points):
        return points / 72.0 * self.dpi

    def draw_image(self, gc, x, y, im):
        pass

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        pass

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform):
        pass

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

    def get_text_width_height_descent(self, s, prop, ismath):
        fontsize = prop.get_size_in_points()
        width = len(s) * fontsize * 0.6
        height = fontsize
        descent = fontsize * 0.2
        return width, height, descent


class FigureCanvasDxf(FigureCanvasBase):
    """
    A canvas to use the renderer. This only implements enough of the
    API to allow the export of DXF to file.
    """

    DXFVERSION = "AC1032"

    def __init__(self, figure, use_ngi_layers=False):
        super().__init__(figure)
        self.use_ngi_layers = use_ngi_layers

    def get_dxf_renderer(self, cleared=False):
        """Get a renderer to use."""
        l, b, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi, self.use_ngi_layers
        try:
            self._lastKey, self.dxf_renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = self._lastKey != key

        if need_new_renderer:
            self.dxf_renderer = RendererDxf(
                w, h, self.figure.dpi, self.DXFVERSION, self.use_ngi_layers
            )
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

    filetypes = FigureCanvasBase.filetypes.copy()
    filetypes["dxf"] = "DXF"

    def print_dxf(self, filename=None, *args, **kwargs):
        """Write out a DXF file."""
        drawing = self.draw()
        if isinstance(filename, StringIO):
            drawing.write(filename)
        else:
            drawing.saveas(filename)

    def get_default_filetype(self):
        return "dxf"


class FigureCanvasDxfNGI(FigureCanvasDxf):
    """NGI-specific DXF canvas with predefined layers"""

    def __init__(self, figure):
        super().__init__(figure, use_ngi_layers=True)


FigureManagerDXF = FigureManagerBase

# Standard names that backend.__init__ is expecting
FigureCanvas = FigureCanvasDxf
