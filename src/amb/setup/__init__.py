"""外部依赖的一键 setup。⛔ 钉死版本，记录实际装到的那个。"""

from amb.setup.install import (
    install,
    install_all,
    snapshot,
    status,
)
from amb.setup.spec import (
    Dependency,
    Installed,
    Kind,
    LOCKFILE,
    REGISTRY,
    SetupError,
    VersionMismatch,
    dependency,
    require_installed,
)

__all__ = [
    "Dependency", "Installed", "Kind", "LOCKFILE", "REGISTRY", "SetupError",
    "VersionMismatch", "dependency", "install", "install_all",
    "require_installed", "snapshot", "status",
]
