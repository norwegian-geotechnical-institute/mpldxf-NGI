import ezdxf

from mpldxf.text_drawing import draw_text_entity


class _GC:
    def get_rgb(self):
        return (0, 0, 0, 1)


class _Prop:
    def get_size_in_points(self):
        return 12


def test_draw_text_entity_falls_back_to_xy_when_mtext_is_none():
    doc = ezdxf.new()
    msp = doc.modelspace()

    draw_text_entity(
        msp,
        _GC(),
        "Axis Label",
        _Prop(),
        0.0,
        mtext=None,
        points_to_pixels=lambda p: p,
        use_fm_layers=False,
        determine_text_layer=lambda *_args, **_kwargs: "0",
        x=10,
        y=20,
    )

    texts = [e for e in msp if e.dxftype() == "TEXT"]
    assert len(texts) == 1
    assert texts[0].dxf.text == "Axis Label"
    assert tuple(texts[0].dxf.insert) == (10.0, 20.0, 0.0)

