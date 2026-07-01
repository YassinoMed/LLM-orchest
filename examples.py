#!/usr/bin/env python3
"""Démo jouable de l'orchestrateur.

Lance ce script pour voir l'orchestrateur router une série de requêtes
représentatives (les exemples de la spec + d'autres cas couvrant chaque
catégorie).

    python examples.py
"""

from __future__ import annotations

from orchestrator import Orchestrator

# Requêtes de démonstration — une par catégorie + cas limites.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("code",        "Peux-tu me montrer comment trier une liste de dictionnaires par une clé en Python ?"),
    ("code",        "Écris une fonction qui additionne deux nombres en JavaScript."),
    ("math",        "Combien font 345 * 678 ?"),
    ("math",        "Résous l'équation 2x + 3 = 11."),
    ("creative",    "Raconte-moi une histoire triste à propos d'un robot qui apprend à aimer."),
    ("creative",    "Écris un poème sur l'automne."),
    ("factual",     "Explique-moi la photosynthèse."),
    ("factual",     "Qu'est-ce que la photosynthèse ?"),
    ("translation", "Traduis cette phrase en espagnol : bonjour tout le monde."),
    ("translation", "Comment dit-on 'merci beaucoup' en japonais ?"),
    ("general",     "Bonjour, comment vas-tu ?"),
    ("general",     "Donne-moi un conseil pour mieux dormir."),
)


def main() -> None:
    orch = Orchestrator()
    print("=" * 70)
    print("Démonstration de l'orchestrateur")
    print("=" * 70)
    for expected, query in EXAMPLES:
        result = orch.route(query)
        ok = "✓" if result.model == expected else f"✗ (attendu: {expected})"
        print(f"\nRequête : {query}")
        print(f"  → {result.to_json(indent=2)}  {ok}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
