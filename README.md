# LLM Orchestrator

Orchestrateur intelligent de modèles de langage. Il **analyse une requête** en
langage naturel, **détermine le modèle spécialisé le plus approprié**, puis la
**reformule en un prompt optimisé** pour ce modèle cible.

```json
{"model": "math", "reduced_prompt": "Calcule 345 * 678."}
```

## Caractéristiques

- **Deux moteurs de routage** au choix :
  - **Heuristique** (par défaut) — règles pondérées 100 % local, aucun appel
    à un LLM, aucune clé API, aucune latence réseau. Idéal pour comprendre le
    mécanisme "from scratch".
  - **Neuronal** — MLP à **12 couches** entraîné sur un dataset synthétique
    (PyTorch + TF-IDF), généralise mieux que des regex sur des variantes de
    formulation.
- **Sortie JSON stricte** conforme à la spec (`{"model", "reduced_prompt"}`).
- **Trois interfaces** : bibliothèque Python réutilisable, CLI, et API HTTP
  (FastAPI + Swagger auto-généré).
- **82 tests** (dont pipeline ML end-to-end), incluant les exemples officiels
  de la spec comme cas de non-régression bloquants.

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

### Mode neuronal (optionnel)

Le mode neuronal nécessite PyTorch et un entraînement préalable (~1-2 min sur
Apple Silicon / GPU). C'est un routeur alternatif ; le mode heuristique reste
disponible sans aucune dépendance ML.

```bash
# 1. Installer les dépendances ML (optionnelles)
pip install torch numpy datasets

# 2. Entraîner le modèle (une fois)
python -m orchestrator.train --epochs 30
# → génère checkpoints/model.pt + checkpoints/tfidf.json

# 3. Utiliser le mode neuronal
python cli.py "Combien font 345 * 678 ?" --router neural
# {"model": "math", "reduced_prompt": "Calcule 345 * 678."}

# Via l'API HTTP
curl -X POST http://localhost:8000/orchestrate \
     -H "Content-Type: application/json" \
     -d '{"query": "Combien font 345 * 678 ?", "router_mode": "neural"}'
```

En bibliothèque :

```python
from orchestrator import Orchestrator

orch = Orchestrator(router_mode="neural")  # charge les checkpoints
print(orch.to_json("Écris une fonction en Python"))
```

> **Fallback automatique** : si le mode neuronal est demandé mais que les
> checkpoints sont absents (entraînement non fait), l'orchestrateur bascule
> silencieusement sur le mode heuristique. Aucun crash.

## Démo

```bash
python examples.py   # route une série de requêtes représentatives
```

## Comment fonctionne le routage

### Mode heuristique (par défaut)

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

### Mode neuronal (MLP 12 couches)

Le routeur neuronal remplace le scoring de mots-clés par un réseau de neurones
entraîné. Le pipeline :

1. **Dataset synthétique** — `data_generation.py` génère ~3 000 requêtes à
   partir des signaux de `signals.py` (mots-clés + templates de phrases),
   chargées dans un `datasets.Dataset` Hugging Face avec splits train/val/test.
2. **TF-IDF** — `tfidf.py` vectorise le texte en NumPy pur (aucune dépendance
   sklearn, robuste). Vocabulaire + IDF sauvés en JSON.
3. **MLP 12 couches** — `neural_model.py` : feedforward avec dropout,
   `input_dim → 512 → 512 → 256 → 256 → 256 → 256 → 128 → 128 → 128 → 64 → 64 → 6`.
   Exactement 12 couches `nn.Linear`, sortie = 6 logits (un par catégorie,
   ordre de `MODEL_NAMES`).
4. **Entraînement** — `train.py` : CrossEntropyLoss + Adam + weight decay +
   early stopping sur la val accuracy. Device auto (MPS/CUDA/CPU).
5. **Inférence** — `NeuralRouter` charge les checkpoints, applique le TF-IDF,
   passe dans le MLP, argmax → catégorie. Seuil de confiance : sous un certain
   seuil, la requête est classée `general` (même logique d'ambiguïté que
   l'heuristique).

```bash
python -m orchestrator.train --help   # toutes les options (epochs, lr, dropout...)
```

Paramètres principaux du script `train` : `--epochs`, `--batch-size`, `--lr`,
`--dropout`, `--weight-decay`, `--max-features`, `--samples-per-cat`,
`--device {auto,cpu,mps,cuda}`.

## Structure du projet

```
LLM-orchastrator/
├── orchestrator/              # package réutilisable
│   ├── __init__.py            # API publique (orchestrator core + neural)
│   ├── models.py              # RoutingResult, MODEL_NAMES
│   ├── signals.py             # ⭐ config des règles de routage (heuristique)
│   ├── router.py              # classification heuristique (scoring)
│   ├── reducer.py             # reformulation du prompt par catégorie
│   ├── orchestrator.py        # assemblage → JSON (mode heuristic OU neural)
│   ├── tfidf.py               # ⭐ TF-IDF en NumPy pur (from scratch)
│   ├── neural_model.py        # ⭐ MLP 12 couches (PyTorch)
│   ├── data_generation.py     # ⭐ dataset synthétique (HF datasets)
│   ├── neural_router.py       # ⭐ routeur neuronal (même interface que Router)
│   └── train.py               # ⭐ script d'entraînement (python -m orchestrator.train)
├── cli.py                     # interface ligne de commande (--router heuristic|neural)
├── server.py                  # API HTTP (FastAPI, router_mode par requête)
├── examples.py                # démo jouable
├── tests/                     # 82 tests pytest (10 skippés sans checkpoints)
├── requirements.txt
├── checkpoints/               # généré par train.py (gitignoré)
├── data/                      # dataset généré (gitignoré)
└── README.md
```

## Tests

```bash
python -m pytest -v
```

Les tests du routeur neuronal (`test_neural_router.py`) sont automatiquement
**ignorés** tant que les checkpoints n'existent pas. Ils s'activent dès que
vous avez lancé `python -m orchestrator.train`. Les tests du pipeline ML
(`test_training.py`) tournent en quelques secondes sans checkpoints (smoke
test end-to-end avec mini-dataset).

Les 3 exemples officiels de la spec sont des cas de non-régression bloquants
(voir `tests/test_orchestrator.py::test_spec_example_routes_to_expected_model`).

## Limitations assumées

**Reformulation du prompt** : elle est **heuristique** (nettoyage + extraction
par templates), pas générative — il n'y a pas de LLM. Cela suffit pour produire
des prompts propres et fidèles dans la grande majorité des cas, mais ne
rivalisera pas avec une reformulation LLM sur des phrases très complexes.
L'architecture est conçue pour qu'un réducteur LLM puisse être branché plus
tard **sans toucher** au routeur ni à l'orchestrateur.

**Routeur neuronal** : le dataset d'entraînement est **synthétique** (généré à
partir des signaux de `signals.py`). Le modèle apprend donc essentiellement à
imiter les règles heuristiques, en généralisant mieux aux variantes de
formulation. Pour un gain réel sur des données humaines, il faudrait remplacer
le dataset synthétique par des requêtes réelles annotées (étape suivante
possible). Par ailleurs, **12 couches** pour ~3 000 exemples est volontairement
surdimensionné (choix demandé) ; l'overfitting est mitigé par dropout, weight
decay et early stopping, mais reste un risque.
