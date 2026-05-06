# Changes

## Version 1.0.6 

_2026-05-06_

Add:
- Nested block export for subplots: one `main_plot` block inserted into modelspace, containing `subplot_n` inserts; each subplot’s geometry is written inside its corresponding `subplot_n` block.
- `backend_dxf.make_figure_canvas(...)` factory to create a configured FigureCanvas class for `matplotlib.backend_bases.register_backend(...)` (supports `use_subplot_blocks` and `use_fm_layers`).

## Version 1.0.5 

_2026-04-07_

Tidy:
- move code from `backend_dxf.py` into other/new files:
  - `rgb_to_dxf(rgb_val)` into `color_utils.py`
  - fm layer logic into `fm_layers`
  - `filter_invalid_coordinates(vertices)` and `is_valid_coordinate(coord)` into `geometry_utils.py`
  - text drawing logic into `text_drawing.py`

## Version 1.0.4

_2026-03-31_

Add:
- Reusable helpers in `tests/conftest.py` 
- Extended test coverage:
  - clipping
  - markers
  - single point plots
  - marker only plots
  - unfilled markers
  - dashed lines
  - contour hatches
  - FM layers
  - FM test routing
  - geo-pattern artists
  - rotated/aligned text
- added xfail tests for known gaps: NaN-separated lines should split into two polylines, and FM gridlines should route to dedicated grid layers

Tidy
- output pngs and xdfs are saved locally in `tests/artifacts`
- removed duplicated test code in `mpldfx/test_backend_ezdxf.py` 


## Version 1.0.3

_2026-02-24_

Add:

- Split up "FM-Grid" in "FM-Grid-Horizontal" and "FM-Grid-Vertical"
- Add support for plotting FM Samples in dxf:
  - Drawing different linetypes (dotted, dashed etc.)
  - Adding better support for drawing datapoints symbolized by objects as circles, triangles, squares etc.

Fix:

- Unclosed hatch boundary-error
- Duplicated hatch-error
- Missing boundary reactor-error
- Infinite number-error

## Version 1.0.2

_2025-12-16_

Fix:

- Update FM-layers logic to better catch "FM-Grid"

## Version 1.0.1

_2025-12-15_

Add:

- "FM-Grid" to the FM-layers

## Version 1.0.0

_2025-11-25_

Fix:

- Updated the FM-layer functionality to support bar plots

## Version 0.9.0

_2025-11-19_

Add:

- Add support for drawing markers in plots

Fix:

- Fixed issue with some entities ending up in wrong FM-layers

## Version 0.8.0

_2025-11-18_

Add:

- Added support for layers for Field Manager method plots

## Version 0.7.0

_2025-01-08_

Fix:

- Fixed bug for plots with only one data point

## Version 0.6.0

_2024-12-12_

Fix:

- Include support for all path collections

## Version 0.5.0

_2024-12-12_

Fix:

- Strip np.nans from line vertices to avoid silent crash when computing the bounding box

## Version 0.4.0

_2024-11-22_

Add:

- CI/CD pipeline for building and testing the project
