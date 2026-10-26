"""Runtime compatibility for FunASR registries in a frozen application.

FunASR records ``inspect.getsourcelines()`` metadata while registering model
classes. PyInstaller stores bytecode in its archive, so source text is not
available and Python 3.11 raises ``OSError``. The source location is diagnostic
metadata only; model execution does not depend on it.
"""

import inspect


_original_getsourcelines = inspect.getsourcelines


def _frozen_safe_getsourcelines(obj):
    try:
        return _original_getsourcelines(obj)
    except (OSError, TypeError):
        return [], 1


inspect.getsourcelines = _frozen_safe_getsourcelines
