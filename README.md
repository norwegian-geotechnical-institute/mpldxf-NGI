# Matplotlib Backend for DXF

## Overview

This is a Matplotlib backend that enables Matplotlib to save figures as DXF drawings. DXF is a drawing format commonly used by Computer-Aided Design (CAD) tools.

This package builds on the `ezdxf` package by Manfred Moitzi:
[ezdxf on Bitbucket](http://bitbucket.org/mozman/ezdxf)

## Installation

The package can be cloned and installed using poetry:

```bash
poetry install
```

## Usage

To use this backend, you first need to register it with Matplotlib:

```python
import matplotlib
from mpldxf import FigureCanvasDxf
matplotlib.backend_bases.register_backend('dxf', FigureCanvasDxf)
```

Then, you can save a figure as a DXF file:
```python
from matplotlib import pyplot as plt
plt.plot(range(10))
plt.savefig('myplot.dxf')
```


## Warning

This package is a work in progress. Not all Matplotlib plot types will render correctly, and text alignment and sizing in particular may require adjustments.
