# Changes

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
