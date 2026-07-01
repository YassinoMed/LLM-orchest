# LLM Orchestrator

Orchestrateur intelligent de modèles de langage. Il **analyse une requête** en
langage naturel, **détermine le modèle spécialisé le plus approprié**, puis la
**reformule en un prompt optimisé** pour ce modèle cible.

```json
{"model": "math", "reduced_prompt": "Calcule 345 * 678."}
```

## Caractéristiques

- **Routage déterministe, 100 % local** — aucun appel à un LLM, aucune clé API,
  aucune latence réseau. Idéal pour comprendre le mécanisme "from scratch".
- **Sortie JSON stricte** conforme à la spec (`{"model", "reduced_prompt"}`).
- **Trois interfaces** : bibliothèque Python réutilisable, CLI, et API HTTP
  (FastAPI + Swagger auto-généré).
- **70 tests** de non-régression, incluant les exemples officiels de la spec.

## Modèles spécialisés

| Modèle        | Usage                                                         |
|---------------|---------------------------------------------------------------|
| `code`        | Génération, explication ou correction de code                 |
| `creative`    | Écriture créative, narration, poésie, scénarios               |
| `factual`     | Questions factuelles, résumés, explications, définitions      |
| `math`        | Résolution de problèmes mathématiques, calculs, formules      |
| `translation` | Traduction entre langues                                      |
| `general`     | Conversation générale, assistance, conseils — **et ambiguïté** |

> Le modèle `general` joue aussi le rôle de filet de sécurité : toute requête
> ambiguë (signal faible ou deux catégories trop proches) y est routée, avec un
> prompt demandant une clarification.

## Installation

```bash
# Depuis la racine du projet
pip install -r requirements.txt   # uniquement pour l'API HTTP + les tests
```

> Le cœur du routage et le CLI n'utilisent **que la bibliothèque standard
> Python** (aucune dépendance). `fastapi`/`uvicorn` ne sont nécessaires que pour
> l'API HTTP ; `pytest` pour les tests.

## Démarrage rapide

### Bibliothèque Python

```python
from orchestrator import Orchestrator

orch = Orchestrator()
print(orch.to_json("Combien font 345 * 678 ?"))
# {"model": "math", "reduced_prompt": "Calcule 345 * 678."}

result = orch.route("Écris un poème sur l'automne")
print(result.model)           # "creative"
print(result.reduced_prompt)  # "Écris un poème sur l'automne."
```

### CLI

```bash
# JSON compact (comportement par défaut, conforme à la spec)
python cli.py "Combien font 345 * 678 ?"
# {"model": "math", "reduced_prompt": "Calcule 345 * 678."}

# Affichage lisible
python cli.py "Écris un poème sur l'automne" --pretty

# Mode interactif
python cli.py -i

# Lister les modèles disponibles
python cli.py --list-models
```

### API HTTP

```bash
uvicorn server:app --reload        # http://localhost:8000
```

Endpoints :

```bash
# Router une requête
curl -X POST http://localhost:8000/orchestrate \
     -H "Content-Type: application/json" \
     -d '{"query": "Combien font 345 * 678 ?"}'
# {"model":"math","reduced_prompt":"Calcule 345 * 678."}

# Lister les modèles
curl http://localhost:8000/models
# {"models":["code","creative","factual","math","translation","general"]}

# Healthcheck
curl http://localhost:8000/
```

Documentation interactive : **http://localhost:8000/docs** (Swagger UI).

## Démo

```bash
python examples.py   # route une série de requêtes représentatives
```

## Comment fonctionne le routage

L'algorithme est volontairement transparent et sans magie :

1. **Normalisation** — minuscules, espaces normalisés (accents conservés).
2. **Scoring** — pour chaque catégorie, on somme le poids des mots-clés trouvés
   (recherche en mots entiers) et des regex qui matchent. Tout est défini dans
   [`orchestrator/signals.py`](orchestrator/signals.py).
3. **Sélection** — la catégorie au score maximal l'emporte, **sauf** si :
   - le score maximal est trop faible (`min_score`), ou
   - les deux meilleures catégories sont trop proches (`margin`).
   Dans ces deux cas (ambiguïté), on retombe sur `general`.

### Améliorer le routage

Tout passe par **un seul fichier** : `orchestrator/signals.py`. Pour mieux
détecter une catégorie, ajoutez des mots-clés ou des regex pondérés :

```python
"math": {
    "keywords": [
        # ...signaux existants...
        ("mon_nouveau_terme", 4),   # poids : 1-2 faible, 3-4 moyen, 5+ fort
    ],
    "patterns": [
        (_p(r"\bma_nouvelle_regex\b"), 5),
    ],
},
```

Les seuils sont paramétrables :

```python
from orchestrator import Router, Orchestrator

router = Router(min_score=4.0, margin=2.0)  # plus strict sur l'ambiguïté
orch = Orchestrator(router=router)
```

## Structure du projet

```
LLM-orchastrator/
├── orchestrator/          # package réutilisable (cœur, sans dépendance)
│   ├── __init__.py        # API publique
│   ├── models.py          # RoutingResult, MODEL_NAMES
│   ├── signals.py         # ⭐ config centrale des règles de routage
│   ├── router.py          # classification heuristique (scoring)
│   ├── reducer.py         # reformulation du prompt par catégorie
│   └── orchestrator.py    # assemblage → JSON
├── cli.py                 # interface ligne de commande
├── server.py              # API HTTP (FastAPI)
├── examples.py            # démo jouable
├── tests/                 # 70 tests pytest
├── requirements.txt
└── README.md
```

## Tests

```bash
python -m pytest -v
```

Les 3 exemples officiels de la spec sont des cas de non-régression bloquants
(voir `tests/test_orchestrator.py::test_spec_example_routes_to_expected_model`).

## Limitations assumées

La reformulation du prompt est **heuristique** (nettoyage + extraction par
templates), pas générative — il n'y a pas de LLM. Cela suffit pour produire des
prompts propres et fidèles dans la grande majorité des cas, mais ne rivalisera
pas avec une reformulation LLM sur des phrases très complexes.

L'architecture est conçue pour qu'un réducteur LLM puisse être branché plus tard
**sans toucher** au routeur ni à l'orchestrateur (le module `reducer.py` est
isolé derrière une interface simple).
# LLM-orchest
