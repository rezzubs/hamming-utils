from systolic.commands import app
from systolic.logging import setup_logging


def main() -> None:
    setup_logging()
    app()


if __name__ == "__main__":
    main()
