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
from numpy.random import random

from mpldxf import backend_dxf


matplotlib.backend_bases.register_backend("dxf", backend_dxf.FigureCanvas)


class DxfBackendTestCase(unittest.TestCase):
    """Tests for the dxf backend."""

    def test_plot_line_with_no_axis(self):
        """Test a simple line-plot command."""
        plt.gca().patch.set_visible(False)
        plt.plot(range(5), [1, 2, 3, 2, 4])
        plt.axis("off")

        outfile = "tests/files/test_plot_line_with_no_axis.dxf"
        plt.close()

        # Load the DXF file and inspect its content
        doc = ezdxf.readfile(outfile)
        modelspace = doc.modelspace()
        entities = list(modelspace)
        assert len(entities) == 2  # 1 line and the bounding box of the plot

    def test_plot_line(self):
        """Test a simple line-plot command."""
        plt.gca().patch.set_visible(False)
        plt.plot(range(3), [1, 2, 3])
        outfile = "tests/files/test_plot_line.dxf"
        plt.savefig(outfile, transparent=True)
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
        outfile = "tests/files/test_boxplot.dxf"
        plt.savefig(outfile)
        plt.close()

    def test_contour(self):
        """Test some contours."""
        x = np.linspace(-5.0, 5.0, 30)
        y = np.linspace(-5.0, 5.0, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        plt.contour(X, Y, Z)
        outfile = "tests/files/test_contour.dxf"
        plt.savefig(outfile)
        plt.close()

    def test_contourf(self):
        """Test some filled contours."""
        x = np.linspace(-5.0, 5.0, 30)
        y = np.linspace(-5.0, 5.0, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        plt.contourf(X, Y, Z)
        outfile = "tests/files/test_contourf.dxf"
        plt.savefig(outfile)
        plt.close()