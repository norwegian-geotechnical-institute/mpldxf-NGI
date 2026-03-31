from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mpldxf import backend_dxf

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DXF_ARTIFACTS_DIR = ARTIFACTS_DIR / "dxf"
PNG_ARTIFACTS_DIR = ARTIFACTS_DIR / "png"


def _register_backend(canvas_cls):
    matplotlib.backend_bases.register_backend("dxf", canvas_cls)


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def export_dxf():
    DXF_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _export(
        fig,
        name,
        *,
        canvas_cls=backend_dxf.FigureCanvas,
        transparent=False,
        save_preview=True,
    ):
        _register_backend(canvas_cls)
        if save_preview:
            fig.savefig(PNG_ARTIFACTS_DIR / f"{name}.png")

        output = DXF_ARTIFACTS_DIR / f"{name}.dxf"
        fig.savefig(output, transparent=transparent)
        return ezdxf.readfile(output)

    return _export


def modelspace_entities(doc):
    return list(doc.modelspace())


def entities_by_type(doc, dxftype):
    return [entity for entity in modelspace_entities(doc) if entity.dxftype() == dxftype]


def entity_count(doc, dxftype):
    return len(entities_by_type(doc, dxftype))


def entity_types(doc):
    return {entity.dxftype() for entity in modelspace_entities(doc)}


def layer_names(doc):
    return {layer.dxf.name for layer in doc.layers}


def entity_layers(doc):
    return {entity.dxf.layer for entity in modelspace_entities(doc)}


def text_values(doc):
    return [entity.dxf.text for entity in entities_by_type(doc, "TEXT")]


def text_layers(doc):
    return {entity.dxf.text: entity.dxf.layer for entity in entities_by_type(doc, "TEXT")}


def block_names(doc):
    return {block.name for block in doc.blocks}


def text_entity(doc, text):
    for entity in entities_by_type(doc, "TEXT"):
        if entity.dxf.text == text:
            return entity
    raise AssertionError(f"Could not find TEXT entity with content {text!r}")
