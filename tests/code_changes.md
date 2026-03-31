## Previous state

Previously, the backend had a very flat DXF export model: 
- it created one DXF drawing, grabbed modelspace, and then every entity the renderer produced was written directly there.

 In practice that meant lines, text, hatches, markers, and later the custom geo-pattern circles all ended up as peer entities in the same space. 
 
 The renderer already used Matplotlib’s group callbacks like open_group("axes") and close_group("axes"), but only for contextual decisions such as layer assignment and patch/text classification. Those groups did not affect where entities were written. So even if a figure had multiple subplots, the backend treated the whole figure as one undifferentiated output stream.

## Changes
The first part of the change was to make subplot boundaries matter structurally. 
I added a small amount of renderer state in _init_drawing() (line 119): 
- rootspace keeps the real DXF modelspace, 
- modelspace becomes the current write target, 
- _layout_stack lets us temporarily switch targets,
- _axes_block_names / _axes_block_refs track subplot blocks

Then in open_group() (line 183), when Matplotlib opens an axes group, we look up the corresponding Axes object, map its position to a SUBPLOT_n block name, insert that block once, push the current target onto the stack, and redirect self.modelspace to that block definition. While that axes is being drawn, all the existing rendering code continues unchanged, but its add_lwpolyline(), add_text(), add_hatch(), and so on now land inside the subplot block instead of root modelspace. When close_group("axes") runs, we pop back to the previous target. Keying subplot blocks by axes position means overlapping axes such as twinx() share the same subplot block rather than creating duplicates.

After that, you asked for one plot block containing subplot sub-blocks, so I added one more layer above that. The renderer now creates a single PLOT_1 block in init_plot_block() (line 139), inserts that block once into true modelspace, and sets the current target to the plot block before the figure is drawn in FigureCanvasDxf.draw() (line 1018). From there, when open_group("axes") fires, each SUBPLOT_n block is inserted into the current target, which is now the plot block instead of root modelspace. So the hierarchy changed from “everything directly in modelspace” to “modelspace contains one PLOT_1 insert, PLOT_1 contains SUBPLOT_n inserts, and each SUBPLOT_n contains that subplot’s actual geometry.”

A subtle but important follow-up was the geo-pattern post-processing. That code runs after figure.draw(renderer) and used to write directly to renderer.modelspace, which would have broken the new hierarchy. I added layout_for_axes() (line 157) so that _draw_geo_pattern_artists() can resolve the proper subplot block for each axes and add those circles there too (backend_dxf.py (line 1024)). That keeps all subplot-owned geometry together, regardless of whether it was emitted during the normal Matplotlib draw pass or as a later backend-specific step.

The tests changed to reflect that structure. Older tests mostly inspected doc.modelspace() directly because that used to be where the entities lived. Once blocks were introduced, that was no longer the right place to look, so the helper in drawn_entities() (line 37) now gathers entities from SUBPLOT_ blocks first and only falls back to modelspace if there are no subplot blocks. The dedicated subplot test was also rewritten to assert the hierarchy you asked for: exactly one PLOT_ block, exactly two SUBPLOT_ blocks, exactly one insert in modelspace, and exactly two inserts inside the plot block (test_backend_ezdxf.py (line 203)).

So in short, the rendering logic itself is mostly unchanged; the main behavioral shift is where the existing drawing commands are routed. Before, all entities were emitted flat into modelspace. Now the same drawing operations are scoped by Matplotlib’s axes groups and organized into a reusable DXF block hierarchy: whole figure as one block, each subplot as a nested block within it.