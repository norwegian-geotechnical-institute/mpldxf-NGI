import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpldxf import backend_dxf
from conftest import (
    block_names,
    entities_by_type,
    entity_count,
    entity_layers,
    entity_types,
    layer_names,
    modelspace_entities,
    text_entity,
    text_layers,
)


@pytest.mark.parametrize(
    ("name", "x_values", "y_values", "x_limits", "y_limits"),
    [
        (
            "plot_line",
            [0, 1, 2],
            [1, 2, 3],
            None,
            None,
        ),
        (
            "plot_with_data_outside_axes",
            list(range(7)),
            [1, 2, 3, 1e5, 5, 6, 7],
            (1, 6),
            (0, 7),
        ),
    ],
)
def test_basic_line_exports_include_polyline_and_text(
    export_dxf, name, x_values, y_values, x_limits, y_limits
):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot(x_values, y_values)

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    doc = export_dxf(fig, name, transparent=True)

    assert entity_types(doc) == {"LWPOLYLINE", "TEXT"}


def test_plot_line_with_no_axis_exports_only_data_polyline(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot(range(7), [1, 2, 3, 2, 4, 6, 7])
    ax.axis("off")

    doc = export_dxf(fig, "plot_line_with_no_axis", transparent=True)

    assert entity_count(doc, "LWPOLYLINE") == 1
    assert len(modelspace_entities(doc)) == 1


def test_plot_with_twin_axis_and_data_outside_axes(export_dxf):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
    ax2.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
    ax1.set_ylim(1, 6)
    ax2.set_ylim(1, 6)

    doc = export_dxf(fig, "plot_with_twin_axis_and_data_outside_axes", transparent=True)

    assert entity_types(doc) == {"LWPOLYLINE", "TEXT"}


def test_fully_clipped_line_exports_no_entities(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([10, 11], [10, 11])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    doc = export_dxf(fig, "plot_fully_outside_axes", transparent=True)

    assert modelspace_entities(doc) == []


def test_single_point_plot_exports_marker_entities(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([1], [2], marker="o")
    ax.axis("off")

    doc = export_dxf(fig, "single_point_plot", transparent=True)

    assert entity_types(doc) == {"LWPOLYLINE", "CIRCLE", "HATCH"}
    assert entity_count(doc, "CIRCLE") == 1
    assert entity_count(doc, "HATCH") == 1


def test_marker_only_plot_exports_only_marker_geometry(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([0, 1, 2], [1, 2, 1], linestyle="None", marker="o")
    ax.axis("off")

    doc = export_dxf(fig, "marker_only_plot", transparent=True)

    assert entity_types(doc) == {"CIRCLE", "HATCH"}
    assert entity_count(doc, "CIRCLE") == 3
    assert entity_count(doc, "HATCH") == 3


def test_dashed_line_exports_dxf_dash_attributes(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([0, 1, 2], [0, 1, 0], linestyle="--")
    ax.axis("off")

    doc = export_dxf(fig, "dashed_line_plot", transparent=True)
    polyline = entities_by_type(doc, "LWPOLYLINE")[0]

    assert polyline.dxf.linetype == "DASHED2"
    assert polyline.dxf.ltscale == 0.2


def test_unfilled_marker_plot_exports_outline_without_hatch(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot(
        [0, 1, 2],
        [1, 2, 1],
        linestyle="None",
        marker="o",
        markerfacecolor="none",
    )
    ax.axis("off")

    doc = export_dxf(fig, "unfilled_marker_plot", transparent=True)

    assert entity_types(doc) == {"CIRCLE"}
    assert entity_count(doc, "CIRCLE") == 3
    assert entity_count(doc, "HATCH") == 0


def test_marker_plot_exports_filled_marker_geometry(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([0, 1, 2], [0, 1, 0], marker="o")

    doc = export_dxf(fig, "plot_with_markers")

    assert entity_count(doc, "CIRCLE") == 3
    assert entity_count(doc, "HATCH") >= 3


def test_boxplot_export_creates_entities(export_dxf):
    fig, ax = plt.subplots()
    data = [
        [1, 2, 5, 6, 7, 8, 10, 11],
        [3, 4, 6, 7, 8, 9, 12, 13],
        [2, 4, 5, 6, 8, 10, 11, 12],
        [3, 5, 6, 7, 9, 10, 12, 13],
    ]
    ax.boxplot(data)

    doc = export_dxf(fig, "boxplot")

    assert entity_count(doc, "LWPOLYLINE") > 0


def test_contour_export_creates_entities(export_dxf):
    fig, ax = plt.subplots()
    x = np.linspace(-5.0, 5.0, 30)
    y = np.linspace(-5.0, 5.0, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax.contour(X, Y, Z)

    doc = export_dxf(fig, "contour")

    assert entity_count(doc, "LWPOLYLINE") > 0


def test_contourf_export_creates_hatches(export_dxf):
    fig, ax = plt.subplots()
    x = np.linspace(-5.0, 5.0, 30)
    y = np.linspace(-5.0, 5.0, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax.contourf(X, Y, Z)

    doc = export_dxf(fig, "contourf")

    assert entity_count(doc, "HATCH") > 0


def test_plot_with_nans_exports_single_line_hotfix(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, np.nan, 5, 6])
    ax.axis("off")

    doc = export_dxf(fig, "plot_with_nans")

    # The current backend strips the NaN and exports one merged polyline.
    # Ideally this should become two separate polylines with a break at the NaN.
    assert entity_count(doc, "LWPOLYLINE") == 1


@pytest.mark.xfail(reason="NaN-separated line segments are still merged into one polyline")
def test_plot_with_nans_should_split_into_two_polylines(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, np.nan, 5, 6])
    ax.axis("off")

    doc = export_dxf(fig, "plot_with_nans_expected_split")

    assert entity_count(doc, "LWPOLYLINE") == 2


def test_fm_canvas_creates_expected_layers(export_dxf):
    fig, ax = plt.subplots()
    ax.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
    ax.set_ylim(0, 7)
    ax.set_xlim(1, 6)

    doc = export_dxf(
        fig,
        "plot_with_fm_layers",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
        transparent=True,
    )

    assert {
        "FM-Frame",
        "FM-Graph",
        "FM-Method",
        "FM-Text",
        "FM-Depth",
        "FM-Value",
        "FM-Location",
    } <= layer_names(doc)


def test_fm_canvas_assigns_entities_to_expected_layers(export_dxf):
    fig, ax = plt.subplots()
    ax.set_title("Location")
    ax.set_xlabel("Value")
    ax.set_ylabel("Depth")
    ax.plot([0, 1, 2], [0, 1, 0])

    doc = export_dxf(
        fig,
        "plot_with_fm_text_layers",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
    )

    texts = text_layers(doc)

    assert texts["Location"] == "FM-Location"
    assert texts["Value"] == "FM-Text"
    assert texts["Depth"] == "FM-Text"
    assert {"FM-Graph", "FM-Location", "FM-Text", "FM-Value", "FM-Depth"} <= entity_layers(
        doc
    )


def test_fm_canvas_routes_tick_labels_to_value_and_depth_layers(export_dxf):
    fig, ax = plt.subplots()
    ax.set_title("Location")
    ax.set_xlabel("Value")
    ax.set_ylabel("Depth")
    ax.plot([0, 1, 2], [0, 1, 0])

    doc = export_dxf(
        fig,
        "plot_with_fm_tick_layers",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
    )

    texts = text_layers(doc)

    assert texts["0.00"] == "FM-Value"
    assert texts["0.0"] == "FM-Depth"
    assert entity_count(doc, "TEXT") > 0
    assert len(entities_by_type(doc, "TEXT")) >= 3


def test_fm_gid_override_routes_artist_to_method_layer(export_dxf):
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1, 2], [0, 1, 0])
    line.set_gid("FM-Method")

    doc = export_dxf(
        fig,
        "fm_gid_override_method_layer",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
        transparent=True,
    )

    layers = [entity.dxf.layer for entity in entities_by_type(doc, "LWPOLYLINE")]

    assert "FM-Method" in layers


def test_fm_gridlines_currently_export_on_graph_layer(export_dxf):
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.grid(True)

    doc = export_dxf(
        fig,
        "fm_gridlines_current_layering",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
    )

    layers = [entity.dxf.layer for entity in entities_by_type(doc, "LWPOLYLINE")]

    assert "FM-Graph" in layers


@pytest.mark.xfail(
    reason="Gridlines are not yet routed to FM-Grid-Horizontal/FM-Grid-Vertical"
)
def test_fm_gridlines_should_export_to_dedicated_grid_layers(export_dxf):
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.grid(True)

    doc = export_dxf(
        fig,
        "fm_gridlines_expected_layering",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
    )

    layers = {entity.dxf.layer for entity in entities_by_type(doc, "LWPOLYLINE")}

    assert "FM-Grid-Horizontal" in layers
    assert "FM-Grid-Vertical" in layers


def test_text_alignment_exports_right_top_anchor(export_dxf):
    fig, ax = plt.subplots()
    ax.text(0.2, 0.3, "RightTop", ha="right", va="top", transform=ax.transAxes)

    doc = export_dxf(fig, "text_alignment_right_top")
    text = text_entity(doc, "RightTop")

    assert text.dxf.rotation == 0
    assert text.dxf.halign == 2
    assert text.dxf.valign == 3
    assert tuple(text.dxf.insert) == tuple(text.dxf.align_point)


def test_rotated_ylabel_exports_vertical_text_alignment(export_dxf):
    fig, ax = plt.subplots()
    ax.set_ylabel("Rotated Label")

    doc = export_dxf(fig, "rotated_ylabel")
    text = text_entity(doc, "Rotated Label")

    assert text.dxf.rotation == 90.0
    assert text.dxf.halign == 1
    assert text.dxf.valign == 1


def test_text_rotation_preserves_center_middle_alignment(export_dxf):
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        "CenterMiddle",
        ha="center",
        va="center",
        rotation=30,
        transform=ax.transAxes,
    )

    doc = export_dxf(fig, "text_rotation_center_middle")
    text = text_entity(doc, "CenterMiddle")

    assert text.dxf.rotation == 30.0
    assert text.dxf.halign == 1
    assert text.dxf.valign == 2


def test_mathtext_is_sanitized_for_dxf_text_export(export_dxf):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, r"$\mathbf{N}\/60$", transform=ax.transAxes)

    doc = export_dxf(fig, "mathtext_sanitized")
    text = text_entity(doc, "N 60")

    assert text.dxf.text == "N 60"


def test_geo_pattern_vertical_circles_adds_circle_entities(export_dxf):
    fig, ax = plt.subplots()
    artist = ax.scatter([1, 2, 3], [2, 3, 4], s=[36, 36, 36])
    ax._geo_pattern_artists = [{"hatch_style": "VERTICAL_CIRCLES", "artists": artist}]

    doc = export_dxf(
        fig,
        "geo_pattern_vertical_circles",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
    )

    assert entity_count(doc, "CIRCLE") == 3
    assert "FM-Graph" in entity_layers(doc)


def test_geo_pattern_vertical_circles_with_dots_exports_only_circle_entries(export_dxf):
    fig, ax = plt.subplots()
    circles = ax.scatter([1, 2], [2, 3], s=[36, 36])
    dots = ax.scatter([1, 2], [2, 3], s=[4, 4])
    ax._geo_pattern_artists = [
        {
            "hatch_style": "VERTICAL_CIRCLES_WITH_DOTS",
            "artists": [("circle", circles), ("dot", dots)],
        }
    ]

    doc = export_dxf(
        fig,
        "geo_pattern_vertical_circles_with_dots",
        canvas_cls=backend_dxf.FigureCanvasDxfFM,
    )

    assert entity_count(doc, "CIRCLE") == 2
    assert "FM-Graph" in entity_layers(doc)


# --- Additions for subplot sub-blocks and robustness ---


def test_use_fm_layers_string_false_does_not_enable_layers():
    renderer = backend_dxf.RendererDxf(
        100,
        100,
        72,
        backend_dxf.FigureCanvasDxf.DXFVERSION,
        use_fm_layers=False
    )
    assert renderer.use_fm_layers is False


def test_subplots_are_written_to_nested_blocks(export_dxf):
    class FigureCanvasDxfBlocks(backend_dxf.FigureCanvasDxf):
        def __init__(self, figure):
            super().__init__(figure, use_subplot_blocks=True)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot([0, 1], [0, 1])
    axs[1].plot([0, 1], [1, 0])

    doc = export_dxf(
        fig,
        "subplot_blocks",
        canvas_cls=FigureCanvasDxfBlocks,
        transparent=True,
        save_preview=False,
    )

    names = block_names(doc)
    assert "main_plot" in names
    assert len([name for name in names if name.startswith("subplot_")]) == 2
    assert len(list(doc.modelspace().query("INSERT"))) == 1
    assert len(list(doc.blocks["main_plot"].query("INSERT"))) == 2


def test_subplots_with_use_subplot_blocks_false_write_to_modelspace(export_dxf):
    class FigureCanvasDxfNoBlocks(backend_dxf.FigureCanvasDxf):
        def __init__(self, figure):
            super().__init__(figure, use_subplot_blocks=False)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot([0, 1], [0, 1])
    axs[1].plot([0, 1], [1, 0])

    doc = export_dxf(
        fig,
        "subplot_no_blocks",
        canvas_cls=FigureCanvasDxfNoBlocks,
        transparent=True,
        save_preview=False,
    )

    names = block_names(doc)
    assert "main_plot" not in names
    assert not any(name.startswith("subplot_") for name in names)
    assert len(list(doc.modelspace().query("INSERT"))) == 0
    assert len(list(doc.modelspace())) > 0


def test_extra_axes_group_warns_and_keeps_target():
    fig, _ax = plt.subplots()
    renderer = backend_dxf.RendererDxf(
        fig.bbox.bounds[2],
        fig.bbox.bounds[3],
        fig.dpi,
        backend_dxf.FigureCanvasDxf.DXFVERSION,
        use_subplot_blocks=True,
    )
    renderer.figure = fig
    renderer.init_main_plot_block()

    renderer.open_group("axes")
    renderer.close_group("axes")

    original_target = renderer.current_write_target
    assert renderer.current_write_target is original_target
    renderer.close_group("axes")
    assert renderer.current_write_target is original_target
