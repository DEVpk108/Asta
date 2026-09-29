from core.computer.controller import ComputerController


class FakeBackend:
    def __init__(self):
        self.calls = []

    def moveTo(self, x, y, duration=0.0):
        self.calls.append(("moveTo", x, y, duration))

    def click(self, **kwargs):
        self.calls.append(("click", dict(kwargs)))

    def write(self, text, interval=0.0):
        self.calls.append(("write", text, interval))

    def press(self, key):
        self.calls.append(("press", key))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))


def test_mouse_and_keyboard_primitives_use_backend():
    backend = FakeBackend()
    controller = ComputerController(backend=backend)

    assert controller.move_mouse(x=100, y=200) == {
        "x": 100,
        "y": 200,
        "moved": True,
    }
    assert controller.click(x=100, y=200) == {
        "x": 100,
        "y": 200,
        "button": "left",
        "clicks": 1,
        "clicked": True,
    }
    assert controller.type_text("hello", paste=False)["typed"] is True
    assert controller.keypress("enter") == {
        "key": "enter",
        "pressed": True,
    }
    assert controller.hotkey(["ctrl", "l"])["pressed"] is True
    assert controller.scroll(3)["scrolled"] is True
    assert controller.wait(seconds=0.0) == {
        "seconds": 0.0,
        "waited": True,
    }

    assert backend.calls == [
        ("moveTo", 100, 200, 0.0),
        (
            "click",
            {
                "x": 100,
                "y": 200,
                "button": "left",
                "clicks": 1,
                "interval": 0.0,
            },
        ),
        ("write", "hello", 0.0),
        ("press", "enter"),
        ("hotkey", ("ctrl", "l")),
        ("scroll", 3),
    ]



def test_click_tool_allows_runtime_grounded_target():
    from core.tools.computer import ComputerClickTool

    definition = ComputerClickTool().definition

    assert definition.input_schema["required"] == []
    assert "target" in definition.input_schema["properties"]
