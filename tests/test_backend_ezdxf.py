import matplotlib.pyplot as plt
import numpy as np

from mpldxf import backend_dxf
from conftest import entity_types, layer_names, modelspace_entities


def test_plot_line_with_no_axis(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot(range(7), [1, 2, 3, 2, 4, 6, 7])
    ax.axis("off")

    doc = export_dxf(fig, "plot_line_with_no_axis", transparent=True)

    assert len(modelspace_entities(doc)) == 1


def test_plot_line(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot(range(3), [1, 2, 3])

    doc = export_dxf(fig, "plot_line", transparent=True)

    assert entity_types(doc) == {"LWPOLYLINE", "TEXT"}


def test_plot_with_data_outside_axes(export_dxf):
    fig, ax = plt.subplots()
    ax.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
    ax.set_ylim(0, 7)
    ax.set_xlim(1, 6)

    doc = export_dxf(fig, "plot_with_data_outside_axes", transparent=True)

    assert entity_types(doc) == {"LWPOLYLINE", "TEXT"}


def test_plot_with_twin_axis_and_data_outside_axes(export_dxf):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
    ax2.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
    ax1.set_ylim(1, 6)
    ax2.set_ylim(1, 6)

    doc = export_dxf(fig, "plot_with_twin_axis_and_data_outside_axes", transparent=True)

    assert entity_types(doc) == {"LWPOLYLINE", "TEXT"}


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

    assert modelspace_entities(doc)


def test_contour_export_creates_entities(export_dxf):
    fig, ax = plt.subplots()
    x = np.linspace(-5.0, 5.0, 30)
    y = np.linspace(-5.0, 5.0, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax.contour(X, Y, Z)

    doc = export_dxf(fig, "contour")

    assert modelspace_entities(doc)


def test_contourf_export_creates_entities(export_dxf):
    fig, ax = plt.subplots()
    x = np.linspace(-5.0, 5.0, 30)
    y = np.linspace(-5.0, 5.0, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax.contourf(X, Y, Z)

    doc = export_dxf(fig, "contourf")

    assert modelspace_entities(doc)


def test_plot_with_nans_exports_single_line_hotfix(export_dxf):
    fig, ax = plt.subplots()
    ax.patch.set_visible(False)
    ax.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, np.nan, 5, 6])
    ax.axis("off")

    doc = export_dxf(fig, "plot_with_nans")

    assert len(modelspace_entities(doc)) == 1


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
