"""Vectoriseur TF-IDF implémenté en NumPy pur (aucune dépendance à sklearn).

Pourquoi une implémentation maison ? Sur cet environnement macOS, le binaire
scipy/sklearn est cassé. Un TF-IDF est par ailleurs suffisamment simple pour
être écrit proprement from scratch, ce qui correspond aussi à l'esprit du
projet.

Formule utilisée (variante "lisse" proche de sklearn) :
    - tf(t, d)  = nombre d'occurrences du terme t dans le document d
    - idf(t)    = ln((1 + N) / (1 + df(t))) + 1   (lissage +1 pour éviter division par zéro)
    - tfidf     = tf * idf, puis chaque ligne est L2-normalisée.

L'objet est sérialisable en JSON (vocabulaire + idf) pour être rechargé au
moment de l'inférence sans réentraîner.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

# Regex de tokenisation : séquences de lettres (accentuées incluses) ou chiffres.
# On garde volontairement la ponctuation hors du vocabulaire.
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)

# Mots vides français + anglais (liste courte, ciblée pour notre domaine).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à",
        "au", "aux", "ce", "ces", "cet", "cette", "mon", "ma", "mes", "ton",
        "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos",
        "leur", "leurs", "je", "tu", "il", "elle", "on", "nous", "vous",
        "ils", "elles", "me", "te", "se", "lui", "y", "en", "dans", "sur",
        "pour", "par", "avec", "sans", "est", "sont", "été", "être", "ai",
        "as", "a", "avons", "avez", "ont", "the", "a", "an", "of", "to",
        "in", "on", "for", "and", "or", "is", "are", "be", "with", "que",
        "qui", "quoi", "dont", "où", "ne", "pas", "plus", "moins", "très",
        "trop", "peu", "bien", "mal", "comme", "si", "quand", "comment",
    }
)


def normalize_accents(text: str) -> str:
    """Convertit les accents en forme ASCII (ex. « café » → « cafe »).

    Cela permet au vocabulaire de regrouper les variantes accentuées/non
    accentuées, fréquentes en français (« écris » vs « ecris »).
    """
    # NFKD décompose les caractères accentués en base + diacritique.
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Tokenise un texte : minuscules, accents normalisés, mots vides retirés.

    La normalisation des accents est volontaire : le français écrit combine
    souvent formes accentuées et non accentuées.
    """
    text = normalize_accents(text.lower())
    tokens = _TOKEN_RE.findall(text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class TfidfVectorizer:
    """Vectoriseur TF-IDF (NumPy pur) avec sauvegarde/chargement JSON.

    Usage::

        vec = TfidfVectorizer(max_features=5000)
        vec.fit(texts)                 # apprend le vocabulaire + IDF
        X = vec.transform(texts)       # (n_docs, vocab_size) float32 L2-normalisé
    """

    def __init__(self, max_features: int = 5000, min_df: int = 1) -> None:
        """
        Args:
            max_features: taille maximale du vocabulaire (les termes les plus
                fréquents sont conservés en cas de dépassement).
            min_df: nombre minimal de documents dans lesquels un terme doit
                apparaître pour être conservé.
        """
        self.max_features = max_features
        self.min_df = min_df
        # Vocabulaire : mot -> indice de colonne.
        self.vocabulary_: dict[str, int] = {}
        # Vecteur IDF, rempli au fit().
        self.idf_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        """Apprend le vocabulaire et les poids IDF à partir d'une liste de textes."""
        # 1. Compter les fréquences de documents (df) et de termes (tf global).
        doc_freq: dict[str, int] = {}
        total_freq: dict[str, int] = {}
        for text in texts:
            tokens = tokenize(text)
            seen = set()
            for tok in tokens:
                total_freq[tok] = total_freq.get(tok, 0) + 1
                if tok not in seen:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1
                    seen.add(tok)

        # 2. Filtrer par min_df.
        kept = {t for t, df in doc_freq.items() if df >= self.min_df}

        # 3. Limiter à max_features (termes les plus fréquents d'abord).
        if len(kept) > self.max_features:
            kept_sorted = sorted(kept, key=lambda t: total_freq[t], reverse=True)
            kept = set(kept_sorted[: self.max_features])

        # 4. Vocabulaire trié pour un ordre déterministe.
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(kept))}

        # 5. IDF lissé : idf(t) = ln((1 + N) / (1 + df(t))) + 1
        n_docs = len(texts)
        idf = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for word, idx in self.vocabulary_.items():
            df = doc_freq[word]
            idf[idx] = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self.idf_ = idf
        return self

    # ------------------------------------------------------------- transform
    def transform(self, texts: list[str]) -> np.ndarray:
        """Convertit une liste de textes en matrice TF-IDF (n_docs, vocab_size)."""
        if self.idf_ is None:
            raise RuntimeError("Le vectoriseur doit être fitté avant transform().")
        n_docs = len(texts)
        vocab_size = len(self.vocabulary_)
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for tok in tokenize(text):
                idx = self.vocabulary_.get(tok)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            for idx, tf in counts.items():
                matrix[row, idx] = tf

        # Multiplier par IDF.
        matrix *= self.idf_[np.newaxis, :]

        # Normalisation L2 par ligne (évite que les textes longs dominent).
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # éviter division par zéro pour les docs vides
        matrix /= norms
        return matrix.astype(np.float32)

    def transform_one(self, text: str) -> np.ndarray:
        """Convertit un texte unique en vecteur TF-IDF (1D, vocab_size)."""
        return self.transform([text])[0]

    # ---------------------------------------------------- sérialisation JSON
    def save(self, path: str | Path) -> None:
        """Sauvegarde le vocabulaire + IDF dans un fichier JSON."""
        if self.idf_ is None:
            raise RuntimeError("Rien à sauvegarder : vectoriseur non fitté.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "vocabulary": self.vocabulary_,
            "idf": self.idf_.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfVectorizer":
        """Recharge un vectoriseur depuis un fichier JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vec = cls(max_features=payload["max_features"], min_df=payload["min_df"])
        vec.vocabulary_ = {k: int(v) for k, v in payload["vocabulary"].items()}
        vec.idf_ = np.asarray(payload["idf"], dtype=np.float64)
        return vec

    # ---------------------------------------------------------------- helper
    @property
    def vocab_size(self) -> int:
        """Taille du vocabulaire (dimension d'entrée du modèle)."""
        return len(self.vocabulary_)
