from __future__ import annotations

import sys


def main() -> int:
    if "--worker" in sys.argv:
        from asrgui.m0.worker import main as worker_main

        return worker_main([arg for arg in sys.argv[1:] if arg != "--worker"])

    from asrgui.m0.ui import main as ui_main

    return ui_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
