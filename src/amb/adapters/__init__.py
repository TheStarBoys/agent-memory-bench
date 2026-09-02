"""被测系统与对照组的接入层。"""

from amb.adapters.registry import (
    CONTROL_ARMS,
    SYSTEMS,
    create,
    names,
    register,
)

__all__ = ["CONTROL_ARMS", "SYSTEMS", "create", "names", "register"]
