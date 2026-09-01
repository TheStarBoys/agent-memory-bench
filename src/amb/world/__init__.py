"""世界：评测器拥有，可复现，被测系统只读。"""

from amb.world.digest import digest
from amb.world.endpoints import WorldServer
from amb.world.manifest import FileSpec, WorldManifest
from amb.world.materialize import materialize, pin_mtimes
from amb.world.mutate import Change, ChangeKind, WorldState

__all__ = [
    "Change", "ChangeKind", "FileSpec", "WorldManifest", "WorldServer", "WorldState",
    "digest", "materialize", "pin_mtimes",
]
