"""
A backend to export DXF using a custom DXF renderer.

This allows saving of DXF figures.

Use as a matplotlib external backend:

  import matplotlib
  matplotlib.use('module://mpldxf.backend_dxf')

or register:

  matplotlib.backend_bases.register_backend('dxf', FigureCanvasDxf)

Based on matplotlib.backends.backend_template.py.
"""

from io import StringIO
import math

from matplotlib.backend_bases import (
    RendererBase,
    FigureCanvasBase,
    GraphicsContextBase,
    FigureManagerBase,
)
from matplotlib.transforms import Affine2D
import numpy as np
from shapely import Point
from shapely.geometry import LineString, Polygon
import ezdxf
from ezdxf.math.clipping import ClippingRect2d

from mpldxf.text_drawing import draw_text_entity
from .color_utils import rgb_to_dxf
from .fm_layers import (
    create_fm_layers,
    determine_element_layer,
    determine_text_layer,
)
from .geometry_utils import filter_invalid_coordinates, is_valid_coordinate

class RendererDxf(RendererBase):
    """
    The renderer handles drawing/rendering operations.
    Renders the drawing using the ``ezdxf`` package with Field Manager layer support.
    """

    def __init__(
        self,
        width,
        height,
        dpi,
        dxfversion,
        use_fm_layers=False,
        use_subplot_blocks=True,
    ):
        RendererBase.__init__(self)
        self.height = height
        self.width = width
        self.dpi = dpi
        self.dxfversion = dxfversion
        self.use_fm_layers = use_fm_layers
        self.use_subplot_blocks = use_subplot_blocks
        self._init_drawing()
        self._groupd = []
        self._group_gids = {}

    def _init_drawing(self):
        """Create a drawing, set some global information and add the layers we need."""
        # setup=True automatically adds all standard AutoCAD linetypes including DASHED, DASHED2, etc.
        drawing = ezdxf.new(dxfversion=self.dxfversion, setup=True)
        modelspace = drawing.modelspace()
        drawing.header["$EXTMIN"] = (0, 0, 0)
        drawing.header["$EXTMAX"] = (self.width, self.height, 0)

        if self.use_fm_layers:
            create_fm_layers(drawing)

        self.drawing = drawing
        self.modelspace = modelspace
        self.current_write_target = modelspace
        self._write_target_stack = []
        self._next_axes_index = 0
        self._axes_block_names = {}
        self._axes_block_refs = set()

    def init_main_plot_block(self):
        if not self.use_subplot_blocks:
            return
        if self.current_write_target is self.modelspace:
            block_name = "main_plot"
            self.drawing.blocks.new(name=block_name)
            # Keep modelspace to a single top-level insert so the whole figure can
            # be reused as one block in downstream CAD workflows.
            self.modelspace.add_blockref(block_name, (0, 0))
            self.current_write_target = self.drawing.blocks[block_name]

    def _get_block_name_for_axes(self, ax):
        # get subplot block name according to axis bounds
        # if subplot does not exist yet, create it

        bounds = tuple(round(value, 6) for value in ax.get_position().bounds)
        if bounds not in self._axes_block_names:
            block_name = "subplot_%d" % (len(self._axes_block_names) + 1)
            self.drawing.blocks.new(name=block_name)
            # Twin/shared axes reuse the same subplot block because they occupy
            # the same figure position.
            self._axes_block_names[bounds] = block_name
        return self._axes_block_names[bounds]

    def block_for_axes(self, ax):
        if not self.use_subplot_blocks:
            return self.modelspace
        return self.drawing.blocks[self._get_block_name_for_axes(ax)]

    def _get_next_axes_for_group(self):
        if not hasattr(self, "figure"):
            return None

        if self._next_axes_index >= len(self.figure.axes):
           #Got more 'axes' draw groups than Axes in the figure; leaving DXF write target unchanged
            return None

        return self.figure.axes[self._next_axes_index]

    def clear(self):
        """Reset the renderer."""
        self._init_drawing()

    def _push_write_target(self):
        # ``current_write_target`` is where we add new DXF entities.
        # It can be modelspace (legacy mode) or a block definition (sub-block mode).
        # When entering an axes-group we temporarily redirect drawing into a subplot
        # block; this stack remembers the previous destination so we can restore it.
        """Save the current entity destination so it can be restored later."""
        self._write_target_stack.append(self.current_write_target)

    def _pop_write_target(self):
        # Restore the last saved write destination. If the stack is empty we no-op,
        # because Matplotlib group callbacks can be noisy and we want to be robust
        # to mismatched open/close sequences.
        """Restore the previous entity destination (no-op if stack is empty)."""
        if self._write_target_stack:
            self.current_write_target = self._write_target_stack.pop()

    def _enter_axes_group(self):
        """
        Enter an "axes" draw group when sub-blocks are enabled.

        Effect:
        - Save the current destination (usually the main plot block) on the stack.
        - Ensure there's a subplot block for this Axes (named by its position).
        - Insert that subplot block *once* into the current plot block.
        - Redirect subsequent drawing so entities land inside the subplot block.

        When the group closes, ``close_group("axes")`` restores the previous
        destination by popping the stack.
        """
        """Redirect drawing into a subplot block for the next Axes, if available."""
        ax = self._get_next_axes_for_group()
        if ax is None:
            return

        self._push_write_target()

        block_name = self._get_block_name_for_axes(ax)
        if block_name not in self._axes_block_refs:
            # Insert each subplot block once into the current plot block, then draw
            # the axes contents into the subplot block definition itself.
            self.current_write_target.add_blockref(block_name, (0, 0))
            self._axes_block_refs.add(block_name)

        # Redirect drawing for everything inside this Axes group.
        self.current_write_target = self.drawing.blocks[block_name]
        # Advance so the next axes-group maps to the next Axes in figure.axes.
        self._next_axes_index += 1

    def open_group(self, s, gid=None):
        """Open a grouping element with label *s*."""
        if self.use_subplot_blocks and s == "axes" and hasattr(self, "figure"):
            # Only Axes groups affect the DXF write target. All other groups are
            # still tracked in ``self._groupd`` for FM layer routing.
            self._enter_axes_group()
        self._groupd.append(s)
        if gid:
            self._group_gids[s] = gid  # Store gid per group name

    def close_group(self, s):
        """Close a grouping element with label *s*."""
        if self._groupd and self._groupd[-1] == s:
            self._groupd.pop()
            # Remove gid for this group
            self._group_gids.pop(s, None)
        # Matplotlib's group callbacks can be noisy; only Axes groups manipulate the
        # write target stack, and we defensively no-op if the stack is empty.
        if s == "axes":
            # Return to whatever destination was active before this axes-group
            # (usually the main plot block or modelspace).
            self._pop_write_target()

    def _determine_element_layer(self):
        """Determine which layer to use based on matplotlib element context."""
        if not self.use_fm_layers:
            return "0"
        return determine_element_layer(
            self._groupd,
            self._group_gids,
        )

    def _determine_text_layer(self, text_content, fontsize):
        """Determine text layer based on matplotlib context and content."""
        if not self.use_fm_layers:
            return "0"
        return determine_text_layer(
            self._groupd,
            self._group_gids,
            text_content,
            fontsize,
        )

    def _analyze_patch_size(self, vertices, gc):
        """Simple shape-based classification of patches"""

        verts = np.array(vertices)
        min_x, min_y = np.min(verts, axis=0)
        max_x, max_y = np.max(verts, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        # Check if patch has hatching
        has_hatch = gc.get_hatch() is not None
        if has_hatch:
            return "FM-Graph"

        # Avoid divide by zero
        if width == 0 or height == 0:
            return "0"

        aspect_ratio = max(width, height) / min(width, height)
        context_str = " ".join(self._groupd).lower() if self._groupd else ""
        in_axes = "axes" in context_str

        # Bar detection
        if in_axes and aspect_ratio < 1000:
            return "FM-Graph"

        # Default small patches -> "0"
        return "0"

    def _get_polyline_attribs(self, gc):
        """Get polyline attributes with correct layer, color, and linetype"""
        attribs = {}
        if self.use_fm_layers:
            layer_name = self._determine_element_layer()
            attribs["layer"] = layer_name
            attribs["color"] = 256  # ByLayer color
        else:
            attribs["color"] = rgb_to_dxf(gc.get_rgb())

        # Handle line style - use DASHED2 for dashed lines with ltscale=0.2
        dashes = gc.get_dashes()
        if dashes is not None:
            offset, dash_list = dashes
            if dash_list is not None and len(dash_list) > 0:
                # Use DASHED2 for non-continuous lines
                attribs["linetype"] = "DASHED2"
                attribs["ltscale"] = 0.2  # Small scale for finer dash pattern
            else:
                attribs["linetype"] = "CONTINUOUS"
        else:
            attribs["linetype"] = "CONTINUOUS"

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
                # strip nans and infinite values
                vertices = filter_invalid_coordinates(vertices)

                cliprect = Polygon(cliprect)
                if len(vertices) == 1:
                    line = Point(vertices[0])
                else:
                    line = LineString(vertices)
                try:
                    intersection = line.intersection(cliprect)
                except:
                    intersection = Polygon()

                # Check if intersection is a multi-part geometry
                if intersection.is_empty:
                    vertices = []
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
                    # Validate coordinates before adding to DXF
                    vertices = filter_invalid_coordinates(vertices)
                    if len(vertices) > 0 and vertices[0][0] != 0:
                        entity = self.current_write_target.add_lwpolyline(
                            points=vertices, close=False, dxfattribs=dxfattribs
                        )
                    else:
                        entity = None
                else:
                    # Filter invalid coordinates from each segment
                    vertices = [
                        filter_invalid_coordinates(points) for points in vertices
                    ]
                    vertices = [v for v in vertices if len(v) > 0]
                    entity = [
                        self.current_write_target.add_lwpolyline(
                            points=points, close=False, dxfattribs=dxfattribs
                        )
                        for points in vertices
                    ]
            return entity

    def _draw_mpl_line2d(self, gc, path, transform):
        self._draw_mpl_lwpoly(gc, path, transform, obj="line2d")

    def _draw_mpl_patch(self, gc, path, transform, rgbFace=None):
        """Draw a matplotlib patch object"""
        vertices = path.transformed(transform).vertices
        layer_name = self._determine_element_layer()

        if layer_name == "PENDING":
            layer_name = self._analyze_patch_size(vertices, gc)

        # Set up DXF attributes
        dxfattribs = {}
        if self.use_fm_layers:
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

        # Fill the patch if needed
        if rgbFace is not None:
            if isinstance(poly, list):
                for pol in poly:
                    hatch = self.current_write_target.add_hatch(color=256, dxfattribs=dxfattribs)
                    hatch.set_solid_fill()
                    hatch.paths.add_polyline_path(
                        pol.get_points(format="xyb"),
                        is_closed=pol.closed,
                    )
            else:
                hatch = self.current_write_target.add_hatch(color=256, dxfattribs=dxfattribs)
                hatch.set_solid_fill()
                hatch.paths.add_polyline_path(
                    poly.get_points(format="xyb"),
                    is_closed=poly.closed,
                )
        self._draw_mpl_hatch(gc, path, transform, pline=poly, patch_layer=layer_name)

    def _draw_mpl_hatch(self, gc, path, transform, pline=None, patch_layer=None):
        """Draw hatching from matplotlib hatch patterns"""
        if gc.get_hatch() is None:
            return

        if patch_layer is not None:
            layer_name = patch_layer
        elif self.use_fm_layers:
            layer_name = "FM-Graph"
        else:
            layer_name = "0"

        if self.use_fm_layers:
            dxfattribs = {
                "layer": layer_name,
                "color": 256,  # ByLayer
            }
        else:
            dxfattribs = {"color": rgb_to_dxf(gc.get_rgb())}

        # find extents and center of the original unclipped parent path
        ext = path.get_extents(transform=transform)
        dx = ext.x1 - ext.x0
        cx = 0.5 * (ext.x1 + ext.x0)
        dy = ext.y1 - ext.y0
        cy = 0.5 * (ext.y1 + ext.y0)

        rows, cols = math.ceil(dy / self.dpi) - 1, math.ceil(dx / self.dpi) - 1
        rgb = gc.get_hatch_color()
        dxfcolor = rgb_to_dxf(rgb)
        hpath = gc.get_hatch_path()

        _transform = Affine2D().translate(-0.5, -0.5).scale(self.dpi).translate(cx, cy)
        hpatht = hpath.transformed(_transform)

        for irow in range(-rows, rows + 1):
            for icol in range(-cols, cols + 1):
                _trans = Affine2D().translate(icol * self.dpi, irow * self.dpi)
                _hpath = hpatht.transformed(_trans)
                _path = _hpath.to_polygons(closed_only=False)

                for vertices in _path:
                    if pline is not None:
                        for pline_obj in pline:
                            if len(vertices) == 2:
                                clippoly = Polygon(pline_obj.vertices())
                                line = LineString(vertices)
                                clipped = line.intersection(clippoly).coords
                            else:
                                clipped = ezdxf.math.clipping.ClippingRect2d(
                                    pline_obj.vertices(), vertices
                                )
                    else:
                        clipped = []

                    if len(clipped) > 0:
                        # Validate coordinates before adding to DXF
                        clipped = filter_invalid_coordinates(clipped)

                    if len(clipped) > 0:
                        if len(vertices) == 2:
                            attrs = {"color": dxfcolor}
                            if self.use_fm_layers:
                                attrs["layer"] = layer_name
                                attrs["color"] = 256
                            self.current_write_target.add_lwpolyline(
                                points=clipped, dxfattribs=attrs
                            )
                        else:
                            hatch_attrs = dxfattribs.copy()
                            hatch = self.current_write_target.add_hatch(
                                color=256, dxfattribs=hatch_attrs
                            )
                            hatch.set_solid_fill()
                            hatch.paths.add_polyline_path(clipped, is_closed=True)

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
        """Path collections might be fills, bars, or method icons"""
        original_groupd = self._groupd.copy()
        self._groupd.append("method_collection")

        # Transform offsets to data coordinates
        if len(offsets):
            transformed_offsets = offset_trans.transform(offsets)
        else:
            transformed_offsets = np.zeros((0, 2))

        # Iterate through each path
        for path_idx, path in enumerate(paths):
            # If we have no offsets, draw once with master transform
            if len(transformed_offsets) == 0:
                if facecolors.size:
                    rgbFace = facecolors[0] if facecolors is not None else None
                else:
                    rgbFace = None
                self._draw_mpl_patch(gc, path, master_transform, rgbFace=rgbFace)
            else:
                # Draw the path at each offset position
                for offset_idx, offset in enumerate(transformed_offsets):
                    # Build the combined transform
                    # Start with master transform and add translation for offset
                    combined_transform = master_transform + Affine2D().translate(
                        offset[0], offset[1]
                    )

                    # If we have specific transforms for each instance, apply them
                    if len(all_transforms):
                        # all_transforms is typically a numpy array of 2x3 or 3x3 matrices
                        transform_idx = offset_idx % len(all_transforms)
                        path_transform_matrix = all_transforms[transform_idx]

                        # Convert matrix to Affine2D if needed
                        if isinstance(path_transform_matrix, np.ndarray):
                            # Create Affine2D from matrix
                            if path_transform_matrix.shape == (3, 3):
                                # 3x3 matrix
                                path_transform = Affine2D(path_transform_matrix)
                            elif path_transform_matrix.shape == (2, 3):
                                # 2x3 matrix - convert to 3x3
                                matrix_3x3 = np.vstack(
                                    [path_transform_matrix, [0, 0, 1]]
                                )
                                path_transform = Affine2D(matrix_3x3)
                            else:
                                path_transform = Affine2D()  # Identity
                        else:
                            # Already a transform object
                            path_transform = path_transform_matrix

                        # Combine: first apply path transform, then the combined transform
                        combined_transform = combined_transform + path_transform

                    # Get face color for this instance
                    if facecolors.size:
                        color_idx = offset_idx % len(facecolors)
                        rgbFace = facecolors[color_idx]
                    else:
                        rgbFace = None

                    self._draw_mpl_patch(gc, path, combined_transform, rgbFace=rgbFace)

        self._groupd = original_groupd

    def draw_path(self, gc, path, transform, rgbFace=None):
        """Draw a Path instance using the given affine transform."""
        if len(self._groupd) > 0:
            if self._groupd[-1] == "patch":
                self._draw_mpl_patch(gc, path, transform, rgbFace)
            elif self._groupd[-1] == "line2d":
                self._draw_mpl_line2d(gc, path, transform)
        else:
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
            return

        vertices = path.transformed(trans).vertices

        if self.use_fm_layers:
            # Check if this is a tick marker - these should go to FM-Frame
            if any("tick" in g.lower() for g in self._groupd):
                layer_name = "FM-Frame"
            elif "line2d" in self._groupd or any(
                "axes" in g.lower() for g in self._groupd
            ):
                layer_name = "FM-Graph"
            else:
                layer_name = self._determine_element_layer()
        else:
            layer_name = "0"

        marker_vertices = marker_path.vertices
        marker_codes = marker_path.codes

        dxfattribs = {}
        if self.use_fm_layers:
            dxfattribs["layer"] = layer_name
            dxfattribs["color"] = 256
        else:
            dxfattribs["color"] = rgb_to_dxf(gc.get_rgb())

        from matplotlib.path import Path as MplPath

        segments = []
        segment_closed = []  # Track which segments should be closed
        current_segment = []

        if marker_codes is not None:
            for i, (vertex, code) in enumerate(zip(marker_vertices, marker_codes)):
                if code == MplPath.MOVETO:
                    if current_segment:
                        segments.append(np.array(current_segment))
                        segment_closed.append(False)  # Previous segment wasn't closed
                    current_segment = [vertex]
                elif code == MplPath.LINETO:
                    current_segment.append(vertex)
                elif code == MplPath.CLOSEPOLY:
                    if current_segment:
                        segments.append(np.array(current_segment))
                        segment_closed.append(True)  # This segment is closed
                        current_segment = []

            if current_segment:
                segments.append(np.array(current_segment))
                segment_closed.append(False)
        else:
            for v in marker_vertices:
                if not is_valid_coordinate(v):
                    if current_segment:
                        segments.append(np.array(current_segment))
                        segment_closed.append(False)
                        current_segment = []
                else:
                    current_segment.append(v)

            if current_segment:
                segments.append(np.array(current_segment))
                segment_closed.append(False)

        for vertex in vertices:
            if not is_valid_coordinate(vertex):
                continue

            dx, dy = vertex

            for i, segment in enumerate(segments):
                if len(segment) == 0:
                    continue

                transformed_segment = marker_trans.transform(segment)
                positioned_segment = transformed_segment + np.array([dx, dy])

                # Validate all coordinates in the positioned segment
                positioned_segment = filter_invalid_coordinates(positioned_segment)

                if len(positioned_segment) >= 2:
                    # Determine if this segment should be closed
                    should_close = (
                        segment_closed[i] if i < len(segment_closed) else False
                    )

                    # Also close if first and last points are very close
                    if not should_close:
                        dist = np.linalg.norm(
                            positioned_segment[0] - positioned_segment[-1]
                        )
                        # For circular markers, be more lenient with closing threshold
                        # Calculate approximate marker size
                        if (
                            len(positioned_segment) > 10
                        ):  # Likely a circular marker with many points
                            marker_extent = np.linalg.norm(
                                np.max(positioned_segment, axis=0)
                                - np.min(positioned_segment, axis=0)
                            )
                            # Close if distance is less than 10% of marker extent
                            if dist < marker_extent * 0.1:
                                should_close = True
                        elif dist < 0.5:  # For simple shapes, use fixed threshold
                            should_close = True

                    polyline = self.current_write_target.add_lwpolyline(
                        points=positioned_segment,
                        close=should_close,
                        dxfattribs=dxfattribs,
                    )

                    # Add fill if marker has face color and shape is closed
                    # Check alpha to avoid filling transparent markers
                    face_alpha = (
                        rgbFace[3] if rgbFace is not None and len(rgbFace) > 3 else 1.0
                    )
                    if should_close and rgbFace is not None and face_alpha > 0:
                        hatch = self.current_write_target.add_hatch(
                            color=256, dxfattribs=dxfattribs
                        )
                        hatch.set_solid_fill()
                        hatch.paths.add_polyline_path(
                            polyline.get_points(format="xyb"),
                            is_closed=True,
                        )

                elif len(positioned_segment) == 1:
                    # Single point - likely a circular marker
                    # Calculate radius from marker transform scale
                    scale_x = np.sqrt(
                        marker_trans.get_matrix()[0, 0] ** 2
                        + marker_trans.get_matrix()[1, 0] ** 2
                    )
                    scale_y = np.sqrt(
                        marker_trans.get_matrix()[0, 1] ** 2
                        + marker_trans.get_matrix()[1, 1] ** 2
                    )
                    radius = (
                        (scale_x + scale_y) / 2.0 * 0.5
                    )  # Average scale as radius, scaled down

                    # Use vertex position (dx, dy) as circle center instead of positioned_segment[0]
                    # This ensures the circle is centered correctly on the data point
                    center = [dx, dy]
                    if is_valid_coordinate(center):
                        circle = self.current_write_target.add_circle(
                            center=center,
                            radius=radius,
                            dxfattribs=dxfattribs,
                        )

                        # Add fill for circle if face color specified and not transparent
                        face_alpha = (
                            rgbFace[3]
                            if rgbFace is not None and len(rgbFace) > 3
                            else 1.0
                        )
                        if rgbFace is not None and face_alpha > 0:
                            hatch = self.current_write_target.add_hatch(
                                color=256, dxfattribs=dxfattribs
                            )
                            hatch.set_solid_fill()
                            # Create circular path for hatch
                            hatch.paths.add_edge_path()
                            edge_path = hatch.paths[-1]
                            edge_path.add_arc(
                                center=center,
                                radius=radius,
                                start_angle=0,
                                end_angle=360,
                            )

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """Draw text with proper layer assignment."""
        draw_text_entity(
            self.current_write_target,
            gc,
            s,
            prop,
            angle,
            mtext,
            self.points_to_pixels,
            self.use_fm_layers,
            self._determine_text_layer,
        )

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

    def __init__(self, figure, use_fm_layers=False, use_subplot_blocks=False):
        super().__init__(figure)
        self.use_fm_layers = use_fm_layers
        self.use_subplot_blocks = use_subplot_blocks
        self._lastKey = None

    def get_dxf_renderer(self, cleared=False):
        """Get a renderer to use."""
        l, b, w, h = self.figure.bbox.bounds
        key = (
            w,
            self.figure.dpi,
        )
        if cleared:
            need_new_renderer = True
        else:
            need_new_renderer = self._lastKey != key

        if need_new_renderer:
            self.dxf_renderer = RendererDxf(
                w,
                h,
                self.figure.dpi,
                self.DXFVERSION,
                self.use_fm_layers,
                use_subplot_blocks=self.use_subplot_blocks,
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
        renderer.clear()
        renderer.figure = self.figure
        renderer.init_main_plot_block()
        self.figure.draw(renderer)

        # After drawing, extract any geo pattern artists that weren't drawn
        # These are custom pattern artists stored in axes._geo_pattern_artists
        self._draw_geo_pattern_artists(renderer)

        return renderer.drawing

    def _draw_geo_pattern_artists(self, renderer):
        """Extract and draw geo pattern artists(circles, dots) for FM plots from axes"""
        for ax in self.figure.axes:
            layout = renderer.block_for_axes(ax)
            if not hasattr(ax, "_geo_pattern_artists"):
                continue

            for layer_info in ax._geo_pattern_artists:
                hatch_style = layer_info.get("hatch_style", "")
                artists = layer_info.get("artists")

                if not artists:
                    continue

                # Set up DXF attributes for geo pattern artists
                # These go to FM-Graph layer (or layer 0 if not using FM layers)
                dxfattribs = {}
                if self.use_fm_layers:
                    dxfattribs["layer"] = "FM-Graph"
                    dxfattribs["color"] = 256  # ByLayer
                else:
                    dxfattribs["color"] = 7  # White/default

                if hatch_style == "VERTICAL_CIRCLES":
                    # artists is a PathCollection with all circles
                    try:
                        offsets = artists.get_offsets()
                        sizes = artists.get_sizes()

                        # Transform offsets to data coordinates
                        offset_transform = artists.get_offset_transform()
                        transformed_offsets = offset_transform.transform(offsets)

                        # Calculate radius from size (matplotlib size is area in points^2)
                        # radius = sqrt(size / pi) * (dpi / 72) to convert to drawing units
                        import math

                        if len(sizes) > 0:
                            size = sizes[0]
                            radius = math.sqrt(size / math.pi) * (renderer.dpi / 72.0)
                        else:
                            radius = 2.0  # Default radius

                        for x, y in transformed_offsets:
                            if is_valid_coordinate([x, y]):
                                layout.add_circle(
                                    center=(float(x), float(y)),
                                    radius=radius,
                                    dxfattribs=dxfattribs,
                                )
                    except Exception:
                        pass

                elif hatch_style == "VERTICAL_CIRCLES_WITH_DOTS":
                    # artists is a list of (type, artist) tuples
                    try:
                        import math

                        for element_type, artist in artists:
                            offsets = artist.get_offsets()
                            sizes = artist.get_sizes()

                            # Transform offsets to data coordinates
                            offset_transform = artist.get_offset_transform()
                            transformed_offsets = offset_transform.transform(offsets)

                            if element_type == "circle":
                                # Calculate radius
                                if len(sizes) > 0:
                                    size = sizes[0]
                                    radius = math.sqrt(size / math.pi) * (
                                        renderer.dpi / 72.0
                                    )
                                else:
                                    radius = 2.0

                                for x, y in transformed_offsets:
                                    if is_valid_coordinate([x, y]):
                                        layout.add_circle(
                                            center=(float(x), float(y)),
                                            radius=radius,
                                            dxfattribs=dxfattribs,
                                        )

                    except Exception:
                        pass

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


class FigureCanvasDxfFM(FigureCanvasDxf):
    """FM-specific DXF canvas with predefined layers"""

    def __init__(self, figure):
        super().__init__(figure, use_fm_layers=True)


def make_figure_canvas(*, use_fm_layers=False, use_subplot_blocks=None):
    """
    Create a FigureCanvas class with fixed options.

    Matplotlib's ``register_backend()`` expects a FigureCanvas *class* and does
    not provide a way to pass extra kwargs at registration time, so upstream
    libraries can use this factory to configure mpldxf behavior.
    """

    class _FigureCanvasDxfConfigured(FigureCanvasDxf):
        def __init__(self, figure):
            super().__init__(
                figure,
                use_fm_layers=use_fm_layers,
                use_subplot_blocks=use_subplot_blocks,
            )

    suffix = "FM" if use_fm_layers else "Default"
    sub = (
        "SubBlocksDefault"
        if use_subplot_blocks is None
        else ("SubBlocksOn" if use_subplot_blocks else "SubBlocksOff")
    )
    _FigureCanvasDxfConfigured.__name__ = f"FigureCanvasDxf{suffix}{sub}"
    return _FigureCanvasDxfConfigured


FigureManagerDXF = FigureManagerBase

# Standard names that backend.__init__ is expecting
FigureCanvas = FigureCanvasDxf
