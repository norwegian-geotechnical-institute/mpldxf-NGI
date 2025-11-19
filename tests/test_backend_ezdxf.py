"""Test the dxf matplotlib backend.

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

import unittest

import ezdxf
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from mpldxf import backend_dxf


matplotlib.backend_bases.register_backend("dxf", backend_dxf.FigureCanvas)
matplotlib.use("Agg")


class TestDxfBackendCase(unittest.TestCase):
    """Tests for the dxf backend."""

    def test_plot_line_with_no_axis(self):
        """Test a simple line-plot command."""
        plt.gca().patch.set_visible(False)
        plt.plot(range(7), [1, 2, 3, 2, 4, 6, 7])
        plt.axis("off")
        plt.savefig("tests/files/test_plot_line_with_no_axis.png")

        try:
            outfile = "tests/files/test_plot_line_with_no_axis.dxf"
            plt.savefig(outfile, transparent=True)
        finally:
            plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        modelspace = doc.modelspace()
        entities = list(modelspace)
        assert len(entities) == 1  # 1 line and the bounding box of the plot

    def test_plot_line(self):
        """Test a simple line-plot command."""
        plt.gca().patch.set_visible(False)
        plt.plot(range(3), [1, 2, 3])
        plt.savefig("tests/files/test_plot_line.png")

        try:
            outfile = "tests/files/test_plot_line.dxf"
            plt.savefig(outfile, transparent=True)
        finally:
            plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        modelspace = doc.modelspace()
        entities = list(modelspace)
        entity_types = set([entity.dxftype() for entity in entities])
        assert entity_types == {"LWPOLYLINE", "TEXT"}

    def test_plot_with_data_outside_axes(self):
        """Test a simple line-plot command with data outside the axes."""
        plt.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
        plt.ylim(0, 7)
        plt.xlim(1, 6)
        plt.savefig("tests/files/test_plot_with_data_outside_axes.png")

        try:
            plt.savefig("tests/files/test_plot_with_data_outside_axes.png")
            outfile = "tests/files/test_plot_with_data_outside_axes.dxf"
            plt.savefig(outfile, transparent=True)
        finally:
            plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        modelspace = doc.modelspace()
        entities = list(modelspace)
        entity_types = set([entity.dxftype() for entity in entities])
        assert entity_types == {"LWPOLYLINE", "TEXT"}

    def test_plot_with_twin_axis_and_data_outside_axes(self):
        """Test a simple line-plot command with data outside the axes."""

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
        ax2.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
        ax1.set_ylim(1, 6)
        ax2.set_ylim(1, 6)
        plt.savefig("tests/files/test_plot_with_twin_axis_and_data_outside_axes.png")

        try:
            plt.savefig(
                "tests/files/test_plot_with_twin_axis_and_data_outside_axes.png"
            )
            outfile = "tests/files/test_plot_with_twin_axis_and_data_outside_axes.dxf"
            plt.savefig(outfile, transparent=True)
        finally:
            plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        modelspace = doc.modelspace()
        entities = list(modelspace)
        entity_types = set([entity.dxftype() for entity in entities])
        assert entity_types == {"LWPOLYLINE", "TEXT"}

    def test_boxplot(self):
        """Test a box-plot."""
        data = [
            [1, 2, 5, 6, 7, 8, 10, 11],
            [3, 4, 6, 7, 8, 9, 12, 13],
            [2, 4, 5, 6, 8, 10, 11, 12],
            [3, 5, 6, 7, 9, 10, 12, 13],
        ]
        plt.boxplot(data)
        plt.savefig("tests/files/test_boxplot.png")

        try:
            outfile = "tests/files/test_boxplot.dxf"
            plt.savefig(outfile)
        finally:
            plt.close()

    def test_contour(self):
        """Test some contours."""
        print("TEST CONTOUR")
        x = np.linspace(-5.0, 5.0, 30)
        y = np.linspace(-5.0, 5.0, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        plt.contour(X, Y, Z)
        plt.savefig("tests/files/test_contour.png")

        try:
            outfile = "tests/files/test_contour.dxf"
            plt.savefig(outfile)
        finally:
            plt.close()

    def test_contourf(self):
        """Test some filled contours."""
        x = np.linspace(-5.0, 5.0, 30)
        y = np.linspace(-5.0, 5.0, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        plt.contourf(X, Y, Z)

        plt.savefig("tests/files/test_contourf.png")

        try:
            outfile = "tests/files/test_contourf.dxf"
            plt.savefig(outfile)

        finally:
            plt.close()

    def test_plot_with_nans(self):
        """Test a plot with NaNs."""
        plt.gca().patch.set_visible(False)
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 2, 3, np.nan, 5, 6]
        plt.plot(x, y)
        plt.axis("off")

        plt.savefig("tests/files/test_plot_with_nans.png")

        try:
            outfile = "tests/files/test_plot_with_nans.dxf"
            plt.savefig(outfile)
        finally:
            plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        modelspace = doc.modelspace()
        entities = list(modelspace)
        assert (
            len(entities) == 1
        )  # ideally we should have two lines (i.e. one broken line), but one interpolated line works as a hotfix

    def test_plot_with_data_with_FM_layers(self):
        matplotlib.backend_bases.register_backend("dxf", backend_dxf.FigureCanvasDxfFM)
        """Test a simple line-plot command with data outside the axes."""
        plt.plot(range(7), [1, 2, 3, 1e5, 5, 6, 7])
        plt.ylim(0, 7)
        plt.xlim(1, 6)

        try:
            outfile = "tests/files/test_plot_with_data_outside_axes.dxf"
            plt.savefig(outfile, transparent=True)
        finally:
            plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        # Get all layers
        layers = doc.layers
        layer_names = [layer.dxf.name for layer in layers]

        expected_layers = {
            "FM-Frame",
            "FM-Graph",
            "FM-Method",
            "FM-Text",
            "FM-Depth",
            "FM-Value",
            "FM-Location",
        }

        for expected_layer in expected_layers:
            assert expected_layer in layer_names, (
                f"Layer {expected_layer} not found in DXF file."
            )
