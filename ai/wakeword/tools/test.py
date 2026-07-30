from ai.wakeword.tools.piper_generator import PiperGenerator

generator = PiperGenerator()

generator.reset_manifest()

generator.generate(
    phrases=[
        "Hey ASTA",
        "Hello ASTA",
        "Wake up ASTA",
    ],
    samples_per_phrase=1000,
    overwrite=False,
)