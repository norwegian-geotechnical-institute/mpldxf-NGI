# Matplotlib Backend for DXF

## Overview

This is a Matplotlib backend that enables Matplotlib to save figures as DXF drawings. DXF is a drawing format commonly used by Computer-Aided Design (CAD) tools.

This package builds on the `ezdxf` package by Manfred Moitzi:
[ezdxf on Bitbucket](http://bitbucket.org/mozman/ezdxf)

## Installation of dependencies

The package dependencies can be installed with uv

```bash
uv sync --upgrade
source .venv/bin/activate
```

## Usage

To use this backend, you first need to register it with Matplotlib:

```python
import matplotlib
from mpldxf import backend_dxf

# Matplotlib registration takes a FigureCanvas *class* (no kwargs), so use the
# factory to configure options like subplot sub-blocks.
FigureCanvas = backend_dxf.make_figure_canvas(use_fm_layers=False, use_subplot_blocks=True)
matplotlib.backend_bases.register_backend("dxf", FigureCanvas)
```

Then, you can save a figure as a DXF file:
```python
from matplotlib import pyplot as plt
plt.plot(range(10))
plt.savefig('myplot.dxf')
```

### Feature flags

`mpldxf` can optionally write each subplot into its own DXF block (nested under a single `main_plot` block) to keep modelspace clean and make downstream CAD reuse easier.

- Default: disabled
- Enable (legacy “write directly to modelspace” behavior): pass `use_subplot_blocks=True` via `backend_dxf.make_figure_canvas(...)`.


## Warning

This package is a work in progress. Not all Matplotlib plot types will render correctly, and text alignment and sizing in particular may require adjustments.
