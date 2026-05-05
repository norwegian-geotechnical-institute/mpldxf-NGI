import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpldxf import backend_dxf


matplotlib.backend_bases.register_backend("dxf", backend_dxf.FigureCanvas)
matplotlib.use("Agg")


def drawn_entities(doc):
    block_entities = []
    for block in doc.blocks:
        if block.name.startswith("SUBPLOT_"):
            block_entities.extend(list(block))
    return block_entities or list(doc.modelspace())


class TestDxfBackendCase(unittest.TestCase):
    """Tests for the dxf backend."""

    def test_subplot_blocks_feature_flag_off_keeps_modelspace(self):
        renderer = backend_dxf.RendererDxf(
            100,
            100,
            72,
            backend_dxf.FigureCanvasDxf.DXFVERSION,
            use_subplot_blocks=False,
        )

        renderer.init_main_plot_block()
        assert "MAIN_PLOT" not in {block.name for block in renderer.drawing.blocks}

        original_target = renderer.current_write_target
        renderer.open_group("axes")
        renderer.close_group("axes")
        assert renderer.current_write_target is original_target

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
        entities = drawn_entities(doc)
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
        entities = drawn_entities(doc)
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
        entities = drawn_entities(doc)
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
        entities = drawn_entities(doc)
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
        entities = drawn_entities(doc)
        assert (
            len(entities) == 1
        )  # ideally we should have two lines (i.e. one broken line), but one interpolated line works as a hotfix

    def test_subplots_are_written_to_nested_blocks(self):
        fig, axs = plt.subplots(1, 2)
        axs[0].plot([0, 1], [0, 1])
        axs[1].plot([0, 1], [1, 0])

        try:
            outfile = "tests/files/test_subplots_blocks.dxf"
            plt.savefig(outfile, transparent=True)
        finally:
            plt.close()

        doc = ezdxf.readfile(outfile)
        plot_blocks = [block for block in doc.blocks if block.name == "MAIN_PLOT"]
        subplot_blocks = [block for block in doc.blocks if block.name.startswith("SUBPLOT_")]

        assert len(plot_blocks) == 1
        assert len(subplot_blocks) == 2
        assert len(list(doc.modelspace().query("INSERT"))) == 1
        assert len(list(plot_blocks[0].query("INSERT"))) == 2

    def test_extra_axes_group_warns_and_keeps_target(self):
        fig, ax = plt.subplots()
        renderer = backend_dxf.RendererDxf(
            fig.bbox.bounds[2],
            fig.bbox.bounds[3],
            fig.dpi,
            backend_dxf.FigureCanvasDxf.DXFVERSION,
        )
        renderer.figure = fig
        renderer.init_main_plot_block()

        renderer.open_group("axes")
        renderer.close_group("axes")

        original_target = renderer.current_write_target
        with self.assertWarnsRegex(RuntimeWarning, "more 'axes' draw groups"):
            renderer.open_group("axes")
        assert renderer.current_write_target is original_target
        renderer.close_group("axes")
        assert renderer.current_write_target is original_target
        plt.close(fig)

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
