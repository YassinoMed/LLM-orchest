"""Configuration centrale des signaux de routage (mots-clés + regex pondérés).

C'est le fichier à éditer pour améliorer la qualité du routage : ajouter des
mots-clés, des langages de programmation, des langues cibles, etc. Aucune autre
partie du code ne contient de règles de classification — tout passe par ici.

Structure d'une catégorie :
    - ``keywords`` : couples ``(terme, poids)``. Un terme peut être un mot simple
      ("python") ou une expression ("peux-tu"). La recherche se fait sur des
      *mots entiers* pour les termes sans espace, et en sous-chaîne sinon.
    - ``patterns`` : couples ``(regex compilée, poids)``. La regex est cherchée
      en n'importe quelle position de la requête.

Les poids sont relatifs : plus élevé = signal plus fort. Quelques repères :
    1-2 : indice faible (mot commun pouvant apparaître dans plusieurs contextes)
    3-4 : indice moyen (spécifique mais ambigu)
    5+  : indice fort (quasi-certitude de catégorie)
"""

from __future__ import annotations

import re

from .models import MODEL_NAMES

# Type d'une entrée de signal : liste de (terme/regex, poids).
KeywordSpec = list[tuple[str, int]]
PatternSpec = list[tuple[re.Pattern, int]]


def _p(pattern: str, flags: int = 0) -> re.Pattern:
    """Compile une regex (raccourci pour alléger la config)."""
    return re.compile(pattern, flags)


# --- Langages de programmation détectables (réutilisés pour le code) ---------
PROGRAMMING_LANGUAGES: tuple[str, ...] = (
    "python", "javascript", "js", "typescript", "ts", "java", "c", "c++",
    "cpp", "c#", "csharp", "go", "golang", "rust", "ruby", "php", "swift",
    "kotlin", "scala", "sql", "bash", "shell", "html", "css", "react",
    "vue", "angular", "node",
)

# --- Langues cibles pour la traduction (réutilisées pour translation) --------
TARGET_LANGUAGES: tuple[str, ...] = (
    "français", "francais", "français", "anglais", "espagnol", "allemand",
    "italien", "portugais", "néerlandais", "neerlandais", "russe", "chinois",
    "japonais", "arabe", "hindi", "coréen", "coreen", "hébreu", "hebreu",
    "turc", "polonais", "suédois", "suedois", "finnois", "grec", "latin",
)


# =============================================================================
# Signaux par catégorie
# =============================================================================

SIGNALS: dict[str, dict[str, object]] = {
    # -------------------------------------------------------------------------
    # CODE : génération, explication, correction de code
    # -------------------------------------------------------------------------
    "code": {
        "keywords": [
            # Verbes / intentions
            ("code", 4), ("coder", 4), ("programmer", 4), ("programmation", 4),
            ("fonction", 3), ("fonctions", 3), ("fonctionne pas", 3),
            ("bug", 3), ("debug", 3), ("déboguer", 3), ("erreur", 2),
            ("compiler", 3), ("compilation", 3), ("boucle", 3), ("boucles", 3),
            ("variable", 3), ("variables", 3), ("classe", 3), ("classes", 3),
            ("tableau", 2), ("array", 3), ("dictionnaire", 3), ("liste", 2),
            ("tri", 3), ("trier", 3), ("algorithme", 4), ("algorithmique", 4),
            ("regex", 4), ("expression régulière", 4), ("expressions régulières", 4),
            ("script", 3), ("snippet", 4), ("api", 3), ("framework", 3),
            ("bibliothèque", 2), ("library", 3), ("package", 2), ("module", 2),
            ("récursivité", 4), ("récursif", 4), ("complexité", 2),
            ("compilateur", 4), ("interpréteur", 4), ("syntaxe", 3),
            ("type", 1), ("types", 1), ("exception", 3), ("exceptions", 3),
            ("test unitaire", 4), ("tests unitaires", 4), ("unittest", 5),
            ("pytest", 5), ("documentation", 1),
            # Extensions / verbes d'action code
            ("implémente", 3), ("implémenter", 3), ("refactor", 4),
            ("refactoriser", 4), ("optimise", 3), ("optimiser", 3),
            # Expressions toutes faites
            ("écris un exemple", 2), ("montre-moi comment", 2),
            ("comment faire en", 3), ("exemple de code", 5),
            # Mots-clés de langages (poids fort car peu ambigus)
            *[(_lang, 5) for _lang in PROGRAMMING_LANGUAGES],
        ],
        "patterns": [
            # Définitions de code typiques
            (_p(r"\bdef\s+\w+\s*\("), 5),              # def foo(
            (_p(r"\bfunction\s+\w+\s*\("), 5),         # function foo(
            (_p(r"\bclass\s+\w+"), 4),                 # class Foo
            (_p(r"\bimport\s+\w+"), 3),                # import os
            (_p(r"\bfrom\s+\w+\s+import\b"), 4),       # from os import
            (_p(r"\bprint\s*\("), 3),                  # print(
            (_p(r"\bconsole\.log\s*\("), 5),           # console.log(
            (_p(r"\bSystem\.out\.print"), 5),          # Java
            (_p(r"\bif\s*\(.*\)\s*\{"), 3),            # if (...) {
            (_p(r"\bfor\s*\(.*\)"), 3),                # for (...)
            (_p(r"=>\s*\{?"), 2),                      # arrow function
            # Affectations / opérations typiques du code
            (_p(r"=\s*\[\s*\]"), 2),                   # = []
            (_p(r"=\s*\{\s*\}"), 2),                   # = {}
            (_p(r"\bvar\s+\w+"), 3),                   # var x
            (_p(r"\blet\s+\w+"), 3),                   # let x
            (_p(r"\bconst\s+\w+"), 3),                 # const x
            (_p(r"\bpublic\s+(class|static|void)"), 5),# Java/C#
            (_p(r"\bSELECT\b.*\bFROM\b"), 4),        # SQL SELECT
        ],
    },

    # -------------------------------------------------------------------------
    # MATH : calculs, équations, formules
    # -------------------------------------------------------------------------
    "math": {
        "keywords": [
            ("calcul", 4), ("calcule", 5), ("calculer", 5), ("calculs", 4),
            ("équation", 5), ("équations", 5), ("resoudre", 5), ("résoudre", 5),
            ("résolution", 5), ("théorème", 5), ("theoreme", 5), ("démonstration", 5),
            ("démontrer", 5), ("prouver", 5), ("preuve", 4),
            ("intégrale", 5), ("intégration", 5), ("dérivée", 5), ("derivee", 5),
            ("dérivation", 5), ("limite", 3), ("fonction mathématique", 5),
            ("matrice", 5), ("matrices", 5), ("vecteur", 5), ("vecteurs", 5),
            ("probabilité", 5), ("probabilités", 5), ("statistique", 4),
            ("statistiques", 4), ("moyenne", 3), ("médiane", 4), ("écart-type", 5),
            ("variance", 4), ("pourcentage", 4), ("proportion", 3),
            ("algèbre", 5), ("algebre", 5), ("géométrie", 5), ("geometrie", 5),
            ("trigonométrie", 5), ("trigonometrie", 5), ("polynôme", 5),
            ("factorisation", 5), ("factoriser", 5), ("développer", 2),
            ("multiplication", 4), ("multiplié", 4), ("multiplier", 4),
            ("division", 3), ("diviser", 4), ("addition", 3), ("soustraction", 3),
            ("racine carrée", 5), ("puissance", 4), ("exposant", 4),
            ("fraction", 4), ("nombre premier", 5), ("ppcm", 5), ("pgcd", 5),
            ("combien font", 5), ("combien ça fait", 5),
        ],
        "patterns": [
            # Expression arithmétique explicite : "12 + 5", "345 * 678", "2 ^ 3"
            (_p(r"\d+\s*[+\-*/x×÷^]\s*\d+"), 6),
            # Fraction : "3/4" (mais pas une date brute)
            (_p(r"\b\d+\s*/\s*\d+\b"), 3),
            # Équation avec inconnue : "2x + 3 = 7", "x^2 - 4 = 0"
            (_p(r"\b\d*\s*[a-z]\s*[+\-*/^=]"), 4),
            (_p(r"\b[a-z]\s*\^?\d*\s*="), 3),
            # Pourcentage explicite
            (_p(r"\d+\s*%"), 4),
            # "2 + 2 = ?"
            (_p(r"=\s*\?"), 3),
            # Opération isolée avec "x" comme opérateur (multiplication)
            (_p(r"\d+\s*x\s*\d+"), 5),
        ],
    },

    # -------------------------------------------------------------------------
    # TRANSLATION : traduction entre langues
    # -------------------------------------------------------------------------
    "translation": {
        "keywords": [
            ("traduis", 6), ("traduire", 6), ("traduction", 6), ("traductions", 6),
            ("traducteur", 6), ("comment dit-on", 5), ("comment on dit", 5),
            ("dire en", 4), ("équivalent en", 4), ("dans la langue", 4),
            ("en quelle langue", 4),
            # Indirection par langue cible mentionnée explicitement
            *[(_lang, 3) for _lang in TARGET_LANGUAGES],
        ],
        "patterns": [
            # "traduis ... en anglais", "traduire en espagnol"
            (_p(r"\btradui(?:s|re|son)\b.*\ben\s+\w+"), 6),
            # "comment dit-on X en anglais"
            (_p(r"\bcomment\s+(?:dit|dit-on|on dit)\b.*\ben\s+\w+"), 6),
            # Structure "... en <langue>"
            (_p(r"\ben\s+(?:français|francais|anglais|espagnol|allemand|italien|"
                r"portugais|néerlandais|neerlandais|russe|chinois|japonais|"
                r"arabe|hindi|coréen|coreen|hébreu|hebreu|turc|polonais|suédois|"
                r"suedois|finnois|grec|latin)\b"), 4),
        ],
    },

    # -------------------------------------------------------------------------
    # CREATIVE : écriture créative, narration, poésie, scénarios
    # -------------------------------------------------------------------------
    "creative": {
        "keywords": [
            ("raconte", 5), ("raconter", 5), ("histoire", 4), ("histoires", 4),
            ("conte", 4), ("récit", 5), ("récits", 5), ("narration", 5),
            ("narratif", 5), ("nouvelle", 2), ("nouvelles", 4), ("roman", 5),
            ("poème", 5), ("poèmes", 5), ("poésie", 5), ("poesie", 5),
            ("poétique", 5), ("haïku", 5), ("sonnet", 5), ("vers", 2),
            ("scénario", 5), ("scénarios", 5), ("screenplay", 5), ("dialogue", 4),
            ("monologue", 4), ("pièce de théâtre", 5), ("fable", 5),
            ("imagine", 4), ("imaginer", 4), ("imagination", 4), ("invente", 4),
            ("inventer", 4), ("crée une histoire", 5), ("écris une histoire", 5),
            ("écris une nouvelle", 5), ("écris un poème", 6), ("écris un récit", 6),
            ("écris un scénario", 6),
            ("chanson", 4), ("paroles", 3), ("lyrics", 4),
            ("métaphore", 3), ("personnage", 3), ("personnages", 3),
            ("intrigue", 4), ("denouement", 4), ("dénouement", 4),
            ("fin triste", 4), ("heureuse", 2), ("émouvant", 3), ("émotion", 2),
            ("épique", 3), ("fantastique", 2), ("conte de fées", 5),
        ],
        "patterns": [
            # "raconte-moi", "écris-moi"
            (_p(r"\b(?:raconte|écris|invente|imagine)-moi\b"), 5),
        ],
    },

    # -------------------------------------------------------------------------
    # FACTUAL : questions factuelles, résumés, explications, définitions
    # -------------------------------------------------------------------------
    "factual": {
        "keywords": [
            ("explique", 4), ("expliquer", 4), ("explication", 4),
            ("définis", 5), ("définir", 5), ("définition", 5), ("définitions", 5),
            ("qu'est-ce que", 5), ("qu'est-ce qu'", 5), ("qu'est ce que", 5),
            ("que signifie", 5), ("que veut dire", 5), ("c'est quoi", 4),
            ("qui est", 3), ("qui était", 3), ("biographie", 5),
            ("résumé", 4), ("résumer", 5), ("résume", 5), ("synthèse", 4),
            ("comment", 2), ("pourquoi", 2), ("pour quelles raisons", 3),
            ("différence entre", 4), ("différences entre", 4),
            ("cause", 2), ("conséquence", 2), ("origine", 2), ("histoire de", 2),
            ("fonctionnement", 3), ("principe", 3), ("mécanisme", 3),
            ("caractéristiques", 3), ("propriétés", 3), ("avantages", 2),
            ("inconvénients", 2), ("exemples de", 2), ("liste de", 2),
            ("comparaison", 3), ("comparer", 3),
            ("quand", 1), ("où", 1), ("où se trouve", 3), ("capitale", 3),
        ],
        "patterns": [
            # Questions directes en "est-ce que" + verbe d'état/connaissance
            (_p(r"\bqu['e ]est-ce que\b"), 4),
            # "C'est quoi X ?"
            (_p(r"\bc['e ]est quoi\b"), 3),
            # "Donne-moi la définition de"
            (_p(r"\b(?:donne|donnez)-moi\s+(?:la\s+)?(?:définition|définitions)\b"), 5),
        ],
    },

    # -------------------------------------------------------------------------
    # GENERAL : conversation générale, assistance, conseils, ambiguïté
    # -------------------------------------------------------------------------
    "general": {
        "keywords": [
            ("aide", 2), ("aider", 2), ("conseil", 3), ("conseils", 3),
            ("conseiller", 3), ("recommandation", 3), ("suggestion", 3),
            ("bonjour", 1), ("salut", 1), ("merci", 1), ("au revoir", 1),
            ("comment vas-tu", 2), ("ça va", 2), ("comment ça va", 2),
            ("opinion", 2), ("avis", 2), ("que penses-tu", 3),
            ("que faire", 2), ("que dois-je faire", 3),
            ("parlons de", 2), ("discuter", 2), ("discussion", 2),
        ],
        # Pas de patterns : "general" est surtout le filet de sécurité.
        "patterns": [],
    },
}


# Vérification de cohérence : toutes les catégories déclarées existent.
_missing = set(SIGNALS) - set(MODEL_NAMES)
_extra = set(MODEL_NAMES) - set(SIGNALS)
if _missing or _extra:
    raise RuntimeError(
        f"Incohérence entre SIGNALS et MODEL_NAMES — manquants: {_missing}, "
        f"en trop: {_extra}"
    )
