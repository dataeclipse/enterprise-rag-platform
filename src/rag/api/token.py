import sys

from rag.api.auth import create_access_token
from rag.config import get_settings


def main() -> None:
    subject = sys.argv[1] if len(sys.argv) > 1 else "admin"
    sys.stdout.write(create_access_token(subject, get_settings().auth) + "\n")


if __name__ == "__main__":
    main()
