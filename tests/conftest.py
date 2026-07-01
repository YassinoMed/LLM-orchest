"""Configuration pytest : ajoute la racine du projet au chemin d'import.

Permet d'écrire ``from orchestrator import ...`` dans les tests sans installer
le package ni manipuler ``sys.path`` manuellement.
"""

import sys
from pathlib import Path

# Remonte d'un niveau (tests/ → racine du projet) pour rendre le package importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
