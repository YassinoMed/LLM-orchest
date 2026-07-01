"""Réduction / reformulation déterministe du prompt.

Sans LLM, on ne peut pas « réécrire » intelligemment une phrase. À la place,
on applique une chaîne de transformations déterministes qui produisent un
prompt court et fidèle à l'intention de l'utilisateur :

    1. Nettoyage commun (politesse, formules, espaces).
    2. Template spécialisé selon la catégorie détectée, qui extrait les
       informations utiles (expression numérique, langue cible, langage…).
    3. Fallback : texte nettoyé si aucun template ne capture l'intention.

> Honnêteté : le rendu reste heuristique. L'interface :func:`reduce` est
> suffisamment isolée pour qu'on puisse brancher un réducteur LLM plus tard
> sans toucher au routeur ni à l'orchestrateur.
"""

from __future__ import annotations

import re

from .signals import PROGRAMMING_LANGUAGES, TARGET_LANGUAGES

# Formules de politesse / remplissage à retirer (insensibles à la casse).
# On conserve le sens, on supprime uniquement le bruit conversationnel.
_FILLER_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b(?:peux[-\s]?tu|peux-tu me|pourrais[-\s]?tu|pourriez[-\s]?vous|"
               r"est[-\s]?ce que tu pourrais|est[-\s]?ce que tu peux|"
               r"est[-\s]?ce que vous pourriez|veux[-\s]?tu bien|s'il te pla[îi]t|"
               r"s'il vous pla[îi]t|stp|stpp|merci d'avance|merci de)\b", re.IGNORECASE),
    re.compile(r"\bje voudrais\b", re.IGNORECASE),
    re.compile(r"\bje veux\b", re.IGNORECASE),
    re.compile(r"\bmontre[-\s]?moi\b", re.IGNORECASE),
    re.compile(r"\bdis[-\s]?moi\b", re.IGNORECASE),
    re.compile(r"\bdonne[-\s]?moi\b", re.IGNORECASE),
)

# Ponctuation de fin à normaliser.
_TRAILING_PUNCT = re.compile(r"[?!.\s]+$")


def _clean_common(query: str) -> str:
    """Nettoyage commun : retire formules de politesse, ponctuation finale et normalise les espaces."""
    text = query
    for pattern in _FILLER_PATTERNS:
        text = pattern.sub("", text)
    # Collage des espaces multiples, ponctuation finale et bouts vides.
    text = re.sub(r"\s+", " ", text).strip(" ,;:?!.")
    return text


def _capitalize_first(text: str) -> str:
    """Met la première lettre en majuscule (utile pour un rendu propre)."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def _ensure_trailing_dot(text: str) -> str:
    """Garantit un point final (sans doublon)."""
    text = text.rstrip(" .")
    return text + "." if text else text


def _detect_language(text: str) -> str | None:
    """Détecte un langage de programmation mentionné dans la requête."""
    low = text.lower()
    for lang in PROGRAMMING_LANGUAGES:
        if re.search(rf"\b{re.escape(lang)}\b", low):
            # Jolies formes d'affichage.
            pretty = {
                "js": "JavaScript", "ts": "TypeScript", "cpp": "C++",
                "csharp": "C#", "golang": "Go", "node": "Node.js",
            }.get(lang, lang.capitalize())
            return pretty
    return None


def _detect_target_language(text: str) -> str | None:
    """Détecte une langue cible de traduction mentionnée dans la requête."""
    low = text.lower()
    for lang in TARGET_LANGUAGES:
        if lang in low:
            return lang
    return None


def _extract_math_expression(text: str) -> str | None:
    """Extrait l'expression arithmétique principale de la requête.

    Ex. "Combien font 345 * 678 ?" → "345 * 678"
    """
    # On cherche d'abord une opération avec opérateurs explicites.
    m = re.search(r"(\d+(?:\s*[+\-*/x×÷^]\s*\d+)+)", text)
    if m:
        expr = m.group(1).strip()
        # Normalisation de "x" et "×" en "*" pour la lisibilité.
        expr = re.sub(r"[x×]", " * ", expr)
        expr = re.sub(r"\s+", " ", expr).strip()
        return expr
    return None


def _strip_accents_word(word: str) -> str:
    """Variante sans accents d'un mot (pour matcher « écris »/« ecris »)."""
    return word.translate(str.maketrans("éèêëàâäîïôöùûüç", "eeeeeaaiioouuuc"))


def _reduce_code(cleaned: str, original: str) -> str:
    """Template pour la catégorie ``code``."""
    lang = _detect_language(original)
    intention = _clean_common(original)
    # On retire le langage détecté (pour éviter la redondance « Python ... python »).
    if lang:
        for variant in {lang.lower(), lang.lower().split()[0]}:
            intention = re.sub(rf"\b{re.escape(variant)}\b", "", intention, flags=re.IGNORECASE)
    # On retire les verbes/pronoms introductifs orphelins (avec et sans accents).
    intro_verbs = ["montrer", "écrire", "ecris", "écris", "écrivez", "coder",
                   "faire", "créer", "crée", "implémenter", "donner", "comment",
                   "donnez", "ecrivez", "implementer", "creer", "cree", "ecrire"]
    pattern_verbs = "|".join(sorted({w for v in intro_verbs for w in (v, _strip_accents_word(v))},
                                    key=len, reverse=True))
    intention = re.sub(rf"\b(?:{pattern_verbs})\b", "", intention, flags=re.IGNORECASE)
    intention = re.sub(r"\b(?:me|te|nous|vous)\b", "", intention, flags=re.IGNORECASE)
    intention = re.sub(r"\s+", " ", intention).strip(" ,;:?!.")
    # Retire une préposition « en » orpheline en fin de chaîne (après retrait du langage).
    intention = re.sub(r"\ben$", "", intention, flags=re.IGNORECASE).strip(" ,;:?!.")
    if not intention:
        intention = "réponds à la demande"
    connector = "pour"  # formulation sûre avec infinitif (« pour trier ... »)
    if lang:
        return f"Écris du code {lang} {connector} {intention.lower()}."
    return f"Écris du code {connector} {intention.lower()}."


def _reduce_math(cleaned: str, original: str) -> str:
    """Template pour la catégorie ``math``."""
    expr = _extract_math_expression(original)
    if expr:
        return f"Calcule {expr}."
    # Pas d'expression arithmétique explicite : on garde le texte nettoyé.
    return _ensure_trailing_dot(_capitalize_first(cleaned))


def _reduce_translation(cleaned: str, original: str) -> str:
    """Template pour la catégorie ``translation``."""
    target = _detect_target_language(original)
    # On retire les mots de la requête de traduction pour isoler le contenu.
    content = re.sub(
        r"\b(?:tradui[rs]s?|traduction|comment dit-on|comment on dit|"
        r"dire en|équivalent en)\b", "", original, flags=re.IGNORECASE
    )
    content = _clean_common(content)
    # On retire la langue cible du contenu pour éviter la redondance.
    if target:
        content = re.sub(re.escape(target), "", content, flags=re.IGNORECASE)
        content = re.sub(r"\ben\b", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s+", " ", content).strip(" ,;:")
    if not content:
        content = "le texte fourni"
    if target:
        return f"Traduis « {content} » en {target}."
    return f"Traduis « {content} »."


def _reduce_creative(cleaned: str, original: str) -> str:
    """Template pour la catégorie ``creative``."""
    # On tente d'identifier le type d'œuvre demandée.
    type_map = {
        "poème": "un poème", "poésie": "un poème", "poesie": "un poème",
        "haïku": "un haïku", "sonnet": "un sonnet",
        "histoire": "une histoire", "conte": "un conte",
        "récit": "un récit", "récits": "un récit", "nouvelle": "une nouvelle",
        "roman": "un roman", "scénario": "un scénario",
        "chanson": "une chanson", "fable": "une fable",
        "dialogue": "un dialogue", "monologue": "un monologue",
    }
    low = original.lower()
    work_type = None
    for keyword, phrase in type_map.items():
        if re.search(rf"\b{re.escape(keyword)}\b", low):
            work_type = phrase
            break
    # Sujet : on retire les verbes d'action créative et le type détecté.
    subject = re.sub(
        r"\b(?:raconte|écris|invente|imagine|crée|donne)-moi\b|"
        r"\b(?:raconte|écris|invente|imagine|crée|donnez|donne)\b",
        "", original, flags=re.IGNORECASE
    )
    subject = _clean_common(subject)
    if work_type:
        # On retire le type du sujet s'il s'y trouve pour éviter la répétition.
        for kw in type_map:
            subject = re.sub(rf"\b{re.escape(kw)}\b", "", subject, flags=re.IGNORECASE)
        subject = re.sub(r"\s+", " ", subject).strip(" ,;:?!.")
    # Retire les articles orphelins en début de sujet (« une triste ... » → « triste ... »).
    subject = re.sub(r"^\s*(?:un|une|des|le|la|les|du|de)\s+(?=\S)", "", subject, flags=re.IGNORECASE).strip()
    if not subject:
        subject = "sur le sujet demandé"
    prefix = f"Écris {work_type}" if work_type else "Écris un texte créatif"
    return f"{prefix} {subject.lower()}".strip() + "."


def _reduce_factual(cleaned: str, original: str) -> str:
    """Template pour la catégorie ``factual``."""
    # On retire les marqueurs interrogatifs pour isoler le sujet.
    subject = re.sub(
        r"\b(?:qu'est-ce que|qu'est ce que|qu'est-ce qu'|c'est quoi|"
        r"que signifie|que veut dire|définis|définir|définition|définitions|"
        r"explique(?:-moi)?|expliquer|explication|qui est|qui était|"
        r"donne(?:-moi)?|donnez(?:-moi)?|la|le|les)\b",
        "", original, flags=re.IGNORECASE
    )
    subject = _clean_common(subject)
    if not subject:
        subject = _clean_common(original)
    return f"Explique : {subject.lower()}." if subject else "Explique le sujet demandé."


def _reduce_general(cleaned: str, original: str) -> str:
    """Template pour la catégorie ``general`` (incluant l'ambiguïté).

    Conformément à la spec : si la demande est ambiguë, le reduced_prompt doit
    demander une clarification.
    """
    base = _clean_common(original)
    if not base:
        return ("Précise ta demande : je n'ai pas pu identifier clairement "
                "l'intention (code, créatif, fait, calcul ou traduction).")
    return (f"{base} — si la demande est ambiguë, demande une clarification "
            f"sur l'intention exacte (code, créatif, fait, calcul ou traduction).")


# Dispatch par catégorie. Chaque fonction retourne une chaîne *déjà finalisée*
# (avec ponctuation correcte).
_REDUCERS: dict[str, object] = {
    "code": _reduce_code,
    "math": _reduce_math,
    "translation": _reduce_translation,
    "creative": _reduce_creative,
    "factual": _reduce_factual,
    "general": _reduce_general,
}


def reduce(query: str, model: str) -> str:
    """Reformule la requête en un prompt réduit adapté à la catégorie ``model``.

    Args:
        query: requête originale de l'utilisateur.
        model: catégorie détectée (l'un des identifiants de :data:`MODEL_NAMES`).

    Returns:
        Un prompt concis, prêt à être envoyé au modèle cible.
    """
    cleaned = _clean_common(query)
    reducer_fn = _REDUCERS.get(model, _reduce_general)
    result = reducer_fn(cleaned, query)
    # Sécurité : on garantit toujours une chaîne propre en sortie.
    # On normalise les espaces, on retire la ponctuation finale en double,
    # puis on s'assure qu'il y a exactement un point final.
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"[.?!]+$", "", result).strip()
    return result + "." if result else result
