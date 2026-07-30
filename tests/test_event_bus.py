from core import EventBus

from .dummy_speech import DummySpeechModule
from .dummy_hud import DummyHUDModule


def main():

    print("\n===== ASTA EVENT BUS TEST =====\n")

    bus = EventBus()

    speech = DummySpeechModule(bus)
    hud = DummyHUDModule(bus)

    print("Sending assistant_response event...\n")

    bus.emit(
        "assistant_response",
        text="Hello, I am A.S.T.A."
    )

    print("\nEvent test complete.")


if __name__ == "__main__":
    main()