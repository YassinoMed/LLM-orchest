"""Génération d'un dataset synthétique pour l'entraînement du routeur neuronal.

Le dataset est construit **à partir des signaux de `signals.py`** (mots-clés et
patterns par catégorie) enrichis de templates de phrases variés. Chaque
exemple = une requête textuelle + un label parmi les 6 catégories.

Le résultat est un objet Hugging Face ``datasets.Dataset`` avec splits
train/val/test, sauvegardable en local via ``save_to_disk()`` ou poussable
sur le Hub via ``push_to_hub()``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

# Import lazy : datasets n'est requis que pour l'entraînement, pas pour le core.
try:
    from datasets import Dataset, DatasetDict
except ImportError as exc:
    raise ImportError(
        "Le package 'datasets' de Hugging Face est requis pour la génération "
        "de données. Installe-le avec : pip install datasets"
    ) from exc

from .models import MODEL_NAMES
from .signals import SIGNALS

# =============================================================================
# Templates de phrases par catégorie
# =============================================================================
# Chaque template contient un placeholder {keyword} remplacé par un mot-clé
# ou une expression tirée des signaux de la catégorie.

TEMPLATES: dict[str, list[str]] = {
    "code": [
        "Comment {keyword} en Python",
        "{keyword} en JavaScript",
        "Écris un {keyword} en Java",
        "J'ai un {keyword}, comment le corriger",
        "Montre-moi un {keyword}",
        "Peux-tu me montrer {keyword}",
        "Comment faire un {keyword} en Python",
        "Je veux {keyword} en C++",
        "{keyword} avec React",
        "Bug dans mon {keyword}",
        "Refactorise ce {keyword}",
        "Optimise le {keyword}",
        "Implémente un {keyword}",
        "Debug mon {keyword}",
    ],
    "math": [
        "{keyword}",
        "Combien font {keyword}",
        "Calcule {keyword}",
        "Résouds {keyword}",
        "Peux-tu résoudre {keyword}",
        "Comment calculer {keyword}",
        "{keyword} étape par étape",
        "Aide-moi avec {keyword}",
        "Je ne comprends pas {keyword}",
        "Explique-moi {keyword}",
    ],
    "creative": [
        "Raconte-moi {keyword}",
        "Écris {keyword}",
        "Invente {keyword}",
        "Imagine {keyword}",
        "Je veux {keyword}",
        "Peux-tu créer {keyword}",
        "Compose {keyword}",
        "Rédige {keyword}",
        "Fais-moi {keyword}",
        "Inspire-toi pour écrire {keyword}",
    ],
    "factual": [
        "Qu'est-ce que {keyword}",
        "Explique {keyword}",
        "Définis {keyword}",
        "Peux-tu expliquer {keyword}",
        "Que signifie {keyword}",
        "Comment fonctionne {keyword}",
        "Parle-moi de {keyword}",
        "Donne-moi des infos sur {keyword}",
        "Quelle est la définition de {keyword}",
        "Résume {keyword}",
        "Quelle est la différence entre {keyword} et d'autres concepts",
        "Qui est lié à {keyword}",
        "Pourquoi {keyword} est important",
    ],
    "translation": [
        "Traduis {keyword}",
        "Comment dit-on {keyword}",
        "Traduis en anglais {keyword}",
        "Traduis en espagnol {keyword}",
        "Traduis en allemand {keyword}",
        "Comment on dit {keyword} en anglais",
        "Peux-tu traduire {keyword}",
        "Donne la traduction de {keyword}",
        "Traduction de {keyword}",
    ],
    "general": [
        "Bonjour, comment ça va",
        "Donne-moi un conseil pour {keyword}",
        "Aide-moi avec {keyword}",
        "Que penses-tu de {keyword}",
        "Parlons de {keyword}",
        "Que faire pour {keyword}",
        "Salut, j'ai besoin d'aide avec {keyword}",
        "Merci pour {keyword}",
        "Quelle est ton opinion sur {keyword}",
        "Peux-tu me conseiller sur {keyword}",
    ],
}

# =============================================================================
# Expressions enrichies par catégorie (pour varier le contenu)
# =============================================================================
# Au-delà des mots-clés de signals.py, on ajoute des expressions complètes
# pour produire des requêtes réalistes.

RICH_EXPRESSIONS: dict[str, list[str]] = {
    "code": [
        "tri d'une liste par ordre alphabétique", "fonction de recherche binaire",
        "parcours d'un arbre binaire", "implémentation d'une file d'attente",
        "algorithme de tri rapide", "connexion à une base de données SQL",
        "requête SQL avec jointure", "fonction récursive fibonacci",
        "classe Python pour gérer des utilisateurs", "API REST avec Flask",
        "boucle for pour parcourir un dictionnaire", "gestion des exceptions try/except",
        "décorateur en Python", "compréhension de liste en Python",
        "module d'envoi d'emails", "parser de JSON en JavaScript",
        "fonction callback en Node.js", "test unitaire avec pytest",
        "pattern observer en Java", "structure de données pile",
        "algorithme Dijkstra", "expression régulière pour valider un email",
        "hash map en Go", "serveur HTTP en Rust",
        "tri à bulles en C", "reverse d'une chaîne de caractères",
        "créer un tableau 2D dynamique", "implémenter une pile LIFO",
        "parser un fichier CSV en Python", "routeur Express en Node.js",
        "effectuer une requête HTTP GET en JavaScript", "read et write sur un fichier",
        "générer un nombre aléatoire entre 1 et 100", "compter les mots dans une phrase",
        "enlever les doublons d'une liste", "fusionner deux listes triées",
        "vérifier si un mot est un palindrome", "implémenter le pattern MVC",
        "connecter une application à MongoDB", "créer un thread en Python",
        "convertir JSON en objet Python", "serialiser un dictionnaire en YAML",
        "faire un CRUD complet avec Django", "implémenter un cache LRU",
    ],
    "math": [
        "2x + 3 = 11", "15% de 200", "345 * 678", "racine carrée de 144",
        "dérivée de x^3", "intégrale de 2x", "pgcd de 48 et 18",
        "élimination de Gauss", "matrice 3x3 inverse", "probabilité de tirer un roi",
        "suite arithmétique", "théorème de Pythagore", "factorielle de 10",
        "équation du second degré x^2 - 5x + 6 = 0", "limite de 1/x quand x tend vers 0",
        "permutation de 5 éléments", "volume d'une sphère",
        "résolution de système linéaire", "polynôme du second degré",
        "aire d'un triangle", "périmètre d'un cercle",
        "simplifier la fraction 12/18", "puissance de 2 à 10",
        "somme des entiers de 1 à 100", "moyenne de 5 notes",
        "coefficient directeur d'une droite", "équation d'une parabole",
        "déterminant d'une matrice 2x2", "combinaison de 3 parmi 10",
        "variance d'une série statistique", "écart-type d'un échantillon",
        "convergence d'une suite", "développement limité de e^x",
        "intégrale double", "théorème des accroissements finis",
        "nombre complexe module et argument", "résoudre x^4 - 1 = 0",
    ],
    "creative": [
        "une histoire de science-fiction sur des robots", "un poème sur la pluie",
        "un conte de fées moderne", "un scénario de film d'horreur",
        "une nouvelle policière", "un dialogue entre un chat et un chien",
        "un monologue d'un personnage seul sur une île", "une fable avec une morale",
        "un haïku sur la neige", "une chanson d'amour",
        "une histoire triste sur un vieil homme", "un récit d'aventure en mer",
        "un poème épique sur la guerre", "une pièce de théâtre comique",
        "une histoire fantastique avec des dragons",
        "un conte pour enfants sur la lune", "un thriller psychologique",
        "une lettre d'amour à un inconnu", "un script de pub radio humoristique",
        "une chronique d'un voyage dans le temps", "un texte sur la solitude nocturne",
        "un récit de survie en montagne", "un conte noir sur un détective",
        "une description d'un paysage en automne", "un journal intime fictif",
        "une parabole sur la confiance", "un portrait robot en prose",
        "une histoire dystopique en 2077", "un dialogue entre deux planètes",
        "un poème sur le passage du temps", "un scénario de pub pour un parfum",
    ],
    "factual": [
        "la photosynthèse", "la relativité restreinte", "le changement climatique",
        "le système immunitaire humain", "la révolution industrielle",
        "la théorie de l'évolution", "le fonctionnement d'un moteur électrique",
        "les causes de la Première Guerre mondiale", "la cryptographie quantique",
        "le cycle de l'eau", "la structure de l'ADN", "les bienfaits du sommeil",
        "la différence entre TCP et UDP", "le principe du blockchain",
        "l'histoire de la musique classique",
        "la conquête de l'espace", "le fonctionnement du cerveau humain",
        "l'invention de l'imprimerie", "la mécanique quantique pour les nuls",
        "les énergies renouvelables", "le deep learning expliqué simplement",
        "l'histoire d'Internet", "comment fonctionne un GPS",
        "le réchauffement climatique et ses conséquences", "le système solaire",
        "la guerre froide résumée", "les droits de l'homme",
        "l'importance de la biodiversité", "la révolution numérique",
        "comment fonctionne une pile à combustible", "les trous noirs",
        "l'histoire de la physique quantique", "la Silicon Valley",
        "le marché boursier expliqué", "les algorithmes de tri comparés",
    ],
    "translation": [
        "bonjour tout le monde en anglais", "merci beaucoup en japonais",
        "je t'aime en espagnol", "bonne nuit en allemand",
        "comment allez-vous en italien", "au revoir en portugais",
        "bon appétit en russe", "joyeux anniversaire en arabe",
        "il fait beau aujourd'hui en chinois", "je suis étudiant en coréen",
        "bienvenue en turc", "je m'appelle en néerlandais",
        "je ne comprends pas en suédois", "combien ça coûte en finlandais",
        "où est la gare en polonais", "je voudrais un café en grec",
        "excusez-moi en hébreu", " rendez-vous à 18h en hindi",
    ],
    "general": [
        "mieux dormir la nuit", "gérer le stress au travail",
        "apprendre un nouveau langage", "organiser son temps",
        "améliorer sa productivité", "faire du sport régulièrement",
        "manger plus sainement", "voyager pas cher",
        "décorer son appartement", "choisir un métier",
        "surmonter la procrastination", "améliorer sa confiance en soi",
        "faire connaissance dans une nouvelle ville", "préparer un entretien d'embauche",
        "gérer un budget limité", "entretenir des relations amicales",
        "apprendre à cuisiner", "développer son esprit critique",
        "réussir ses études", "trouver son équilibre vie pro/perso",
    ],
}


# =============================================================================
# Paraphraseurs (augmentations simples pour la diversité)
# =============================================================================
# Appliqués aléatoirement pendant la génération pour varier les formulations.

_PARAPHRASE_PREFIXES: list[str] = [
    "", "", "",  # majoritairement sans préfixe (naturel)
    "peux-tu ", "pourrais-tu ", "est-ce que tu peux ", "je voudrais ",
    "j'aimerais ", "s'il te plaît, ", "help, ", "dis-moi ",
    "explique-moi ", "donne-moi ", "montre-moi ", "comment faire ",
    "je n'arrive pas à ", "aurais-tu une idée pour ",
]

_PARAPHRASE_SUFFIXES: list[str] = [
    "", "", "",  # majoritairement sans suffixe
    " ?", " !", ".", " stp", " s'il te plaît",
    " merci", " si possible", " en détail",
    " rapidement", " simplement", " pour un débutant",
]


def _apply_paraphrase(text: str) -> str:
    """Ajoute aléatoirement un préfixe/suffixe pour diversifier."""
    if random.random() < 0.4:  # 40% du temps, ajouter un préfixe
        prefix = random.choice(_PARAPHRASE_PREFIXES)
        text = prefix + text
    if random.random() < 0.3:  # 30% du temps, ajouter un suffixe
        suffix = random.choice(_PARAPHRASE_SUFFIXES)
        text = text + suffix
    return text


def _apply_case_variation(text: str) -> str:
    """Applique aléatoirement une variation de casse (30% du temps)."""
    r = random.random()
    if r < 0.1:
        return text.upper()  # tout majuscules (exclamation simulée)
    elif r < 0.15:
        return text.capitalize()  # première lettre majuscule
    return text  # minuscules (défaut)


def _extract_keywords(signals_dict: dict) -> list[str]:
    """Extrait la liste des mots-clés depuis la config signals d'une catégorie."""
    keywords = []
    for entry in signals_dict.get("keywords", []):
        keyword = entry[0]  # (terme, poids) → terme
        # On ne garde que les mots-clés d'au moins 2 mots (les expressions)
        # ou les mots simples assez longs/évocateurs.
        if " " in keyword or len(keyword) >= 3:
            keywords.append(keyword)
    return keywords


def generate_dataset(
    seed: int = 42,
    samples_per_category: int = 500,
    extra_variations: int = 200,
) -> DatasetDict:
    """Génère un dataset HF avec des requêtes synthétiques par catégorie.

    Pour chaque catégorie :
        - ``samples_per_category`` exemples générés à partir des templates
          remplis avec les mots-clés de ``signals.py``.
        - ``extra_variations`` exemples générés à partir des expressions
          enrichies (phrases complètes réalistes).

    Args:
        seed: graine aléatoire pour la reproductibilité.
        samples_per_category: nombre d'exemples par catégorie (templates).
        extra_variations: nombre d'exemples supplémentaires par catégorie
            (expressions enrichies).

    Returns:
        Un ``DatasetDict`` avec les splits ``train`` (80%), ``val`` (10%),
        ``test`` (10%), chacun avec les colonnes ``text`` et ``label``.
    """
    random.seed(seed)
    all_texts: list[str] = []
    all_labels: list[int] = []

    for cat_idx, cat_name in enumerate(MODEL_NAMES):
        signals_dict = SIGNALS.get(cat_name, {})
        keywords = _extract_keywords(signals_dict)
        cat_templates = TEMPLATES.get(cat_name, ["{keyword}"])
        cat_expressions = RICH_EXPRESSIONS.get(cat_name, [])

        # --- Variations template + mots-clés ---
        for _ in range(samples_per_category):
            template = random.choice(cat_templates)
            kw = random.choice(keywords) if keywords else "quelque chose"
            text = template.format(keyword=kw)
            # Paraphrase + variation de casse pour la diversité.
            text = _apply_paraphrase(text)
            text = _apply_case_variation(text)
            all_texts.append(text)
            all_labels.append(cat_idx)

        # --- Variations enrichies ---
        for _ in range(extra_variations):
            if cat_expressions:
                expr = random.choice(cat_expressions)
                text = _apply_paraphrase(expr)
                text = _apply_case_variation(text)
                all_texts.append(text)
                all_labels.append(cat_idx)

    # --- Mélange global (important : ne pas avoir toutes les catégories à la suite) ---
    paired = list(zip(all_texts, all_labels))
    random.shuffle(paired)
    all_texts = [p[0] for p in paired]
    all_labels = [p[1] for p in paired]

    # --- Construction du Dataset ---
    ds = Dataset.from_dict({"text": all_texts, "label": all_labels})

    # Split 80/10/10.
    splits = ds.train_test_split(test_size=0.2, seed=seed)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=seed)

    return DatasetDict({
        "train": splits["train"],
        "val": test_val["train"],
        "test": test_val["test"],
    })


def save_dataset(dataset: DatasetDict, path: str | Path) -> None:
    """Sauvegarde le dataset en local."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    print(f"Dataset sauvegardé dans {path} "
          f"(train={len(dataset['train'])}, "
          f"val={len(dataset['val'])}, "
          f"test={len(dataset['test'])})")


def load_dataset(path: str | Path) -> DatasetDict:
    """Recharge un dataset depuis le disque."""
    return DatasetDict.load_from_disk(str(path))


def push_to_hub(dataset: DatasetDict, repo_name: str, **kwargs) -> None:
    """Pousse le dataset sur Hugging Face Hub."""
    dataset.push_to_hub(repo_name, **kwargs)
    print(f"Dataset poussé sur HF Hub : https://huggingface.co/datasets/{repo_name}")
