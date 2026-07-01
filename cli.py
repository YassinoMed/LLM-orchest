#!/usr/bin/env python3
"""Interface en ligne de commande de l'orchestrateur.

Usage :

    # Routage unique → JSON compact (comportement par défaut, conforme à la spec)
    python cli.py "Combien font 345 * 678 ?"

    # Affichage lisible (humain)
    python cli.py "..." --pretty

    # Mode interactif (boucle de requêtes)
    python cli.py -i

    # Lister les modèles disponibles
    python cli.py --list-models
"""

from __future__ import annotations

import argparse
import sys

from orchestrator import MODEL_NAMES, Orchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Orchestrateur de modèles de langage : route une requête "
                    "vers le modèle spécialisé le plus approprié.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Requête en langage naturel à router.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Mode interactif : lit les requêtes sur l'entrée standard jusqu'à "
             "EOF (Ctrl-D) ou la commande 'quit'/'exit'.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Affiche le résultat sous une forme lisible (humain) au lieu du "
             "JSON compact.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Affiche la liste des modèles spécialisés disponibles puis quitte.",
    )
    return parser


def _format_pretty(result) -> str:
    """Représentation lisible d'un RoutingResult."""
    return (
        f"Modèle cible   : {result.model}\n"
        f"Prompt réduit  : {result.reduced_prompt}"
    )


def _handle_query(orch: Orchestrator, query: str, *, pretty: bool) -> int:
    """Traite une requête unique et imprime le résultat. Retourne un code de sortie."""
    result = orch.route(query)
    if pretty:
        print(_format_pretty(result))
    else:
        # JSON strict et compact (aucun autre texte), conforme à la spec.
        print(result.to_json())
    return 0


def _run_interactive(orch: Orchestrator, *, pretty: bool) -> int:
    """Boucle interactive de requêtes."""
    print("Mode interactif — tapez une requête (ou 'quit'/'exit' pour quitter).",
          file=sys.stderr)
    while True:
        try:
            sys.stderr.write("\n> ")
            sys.stderr.flush()
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            print()  # saut de ligne propre après Ctrl-D/Ctrl-C
            return 0
        if not line:
            return 0  # EOF
        query = line.strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", ":q"}:
            return 0
        _handle_query(orch, query, pretty=pretty)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print("Modèles spécialisés disponibles :")
        for name in MODEL_NAMES:
            print(f"  - {name}")
        return 0

    orch = Orchestrator()

    if args.interactive:
        return _run_interactive(orch, pretty=args.pretty)

    if not args.query:
        # Sans requête ni mode interactif : on affiche l'aide.
        parser.print_help(sys.stderr)
        return 2

    return _handle_query(orch, args.query, pretty=args.pretty)


if __name__ == "__main__":
    raise SystemExit(main())
