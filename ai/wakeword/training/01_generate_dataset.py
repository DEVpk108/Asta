from ai.wakeword.tools.piper_generator import PiperGenerator


def main():

    generator = PiperGenerator()

    generator.generate(
        phrases=[
            "Hey ASTA",
            "Hello ASTA",
            "Wake up ASTA",
        ],
        samples_per_phrase=1000,
        overwrite=False,
    )


if __name__ == "__main__":
    main()