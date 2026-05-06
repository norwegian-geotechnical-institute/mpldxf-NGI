import matplotlib.pyplot as plt

from mpldxf import backend_dxf

from conftest import block_names


def test_twinx_reuses_single_subplot_block(export_dxf):
    class FigureCanvasDxfBlocks(backend_dxf.FigureCanvasDxf):
        def __init__(self, figure):
            super().__init__(figure, use_subplot_blocks=True)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot([0, 1], [0, 1])
    ax2.plot([0, 1], [1, 0])

    doc = export_dxf(
        fig,
        "subblocks_twinx_reuse",
        canvas_cls=FigureCanvasDxfBlocks,
        transparent=True,
        save_preview=False,
    )

    # Both Axes occupy the same position, so we should only have one subplot block.
    names = block_names(doc)
    assert "main_plot" in names
    subplot_blocks = [name for name in names if name.startswith("subplot_")]
    assert len(subplot_blocks) == 1

    plot_block = doc.blocks["main_plot"]
    # And the plot block should only insert that subplot block once.
    assert len(list(plot_block.query("INSERT"))) == 1


def test_extra_axes_groups_warn_and_do_not_push_write_target_stack():
    fig, ax = plt.subplots()
    renderer = backend_dxf.RendererDxf(
        fig.bbox.bounds[2],
        fig.bbox.bounds[3],
        fig.dpi,
        backend_dxf.FigureCanvasDxf.DXFVERSION,
        use_subplot_blocks=True,
    )
    renderer.figure = fig
    renderer.init_main_plot_block()

    assert renderer._next_axes_index == 0
    assert len(renderer._write_target_stack) == 0

    renderer.open_group("axes")
    renderer.close_group("axes")
    assert renderer._next_axes_index == 1
    assert len(renderer._write_target_stack) == 0

    # Second axes group with only 1 Axes should warn and not touch the stack.
    assert len(renderer._write_target_stack) == 0
    renderer.close_group("axes")
    assert len(renderer._write_target_stack) == 0

    plt.close(fig)
