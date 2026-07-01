"""Tokenizer word-level from scratch pour le Transformer.

Tokenisation par mots (regex), avec vocabulaire appris sur les données
d'entraînement. Gère le padding et les mots inconnus via des tokens spéciaux.

Interface compatible avec l'ancien ``TfidfVectorizer`` pour minimiser les
changements dans le reste du pipeline :

    - ``fit(texts)``       → construit le vocabulaire
    - ``transform(texts)`` → token ids (batch, max_seq_len) + attention masks
    - ``transform_one(text)`` → token ids (seq_len,) pour le routeur
    - ``encode(text)``     → (input_ids, attention_mask) — helper pratique
    - ``save(path)`` / ``load(path)`` → JSON

Pas de dépendance externe (stdlib regex + numpy).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

# Regex de tokenisation : séquences de lettres accentuées + chiffres + ponctuation interne.
# On sépare la ponctuation forte (en début/fin de mot) en tokens individuels
# pour enrichir le vocabulaire et améliorer la représentation.
_WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[''][A-Za-zÀ-ÖØ-öø-ÿ]+)*"
    r"|[^\sA-Za-zÀ-ÖØ-öø-ÿ0-9]"  # ponctuation isolée
    r"|\d+(?:[.,]\d+)?",           # nombres (avec virgule/point)
    re.UNICODE,
)


def _normalize(text: str) -> str:
    """Normalisation : minuscules + accents décomposés (NFKD) puis recombinaison.

    Ex. « café » → « café » (conservé), « Café » → « café ».
    On ne supprime pas les accents, on ne fait que la casse.
    """
    return text.lower().strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenise un texte en mots et ponctuation.

    Returns:
        Liste de tokens (mots en minuscules, ponctuation isolée, nombres).
    """
    text = _normalize(text)
    return _WORD_RE.findall(text)


# Tokens spéciaux
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


class WordTokenizer:
    """Tokenizer word-level avec padding et gestion des mots inconnus.

    Usage::

        tok = WordTokenizer(max_vocab_size=10000, max_seq_len=64)
        tok.fit(texts)
        input_ids, mask = tok.encode("Combien font 345 * 678 ?")

    Args:
        max_vocab_size: taille maximale du vocabulaire (termes les plus
            fréquents conservés, hors tokens spéciaux).
        min_freq: fréquence minimale d'un mot pour entrer dans le vocabulaire.
        max_seq_len: longueur maximale des séquences (troncation/padding).
    """

    def __init__(
        self,
        max_vocab_size: int = 10000,
        min_freq: int = 2,
        max_seq_len: int = 64,
    ) -> None:
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len

        # Vocabulaire : word -> id. Les tokens spéciaux sont fixes.
        self.vocabulary_: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
        }
        self.word_counts_: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        """Taille totale du vocabulaire (incluant les tokens spéciaux)."""
        return len(self.vocabulary_)

    # ------------------------------------------------------------------ fit
    def fit(self, texts: list[str]) -> "WordTokenizer":
        """Apprend le vocabulaire à partir d'une liste de textes."""
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize_text(text):
                counts[token] = counts.get(token, 0) + 1

        # Filtrer par fréquence minimale.
        kept = {t for t, c in counts.items() if c >= self.min_freq}

        # Limiter aux max_vocab_size termes les plus fréquents
        # (hors tokens spéciaux déjà réservés).
        if len(kept) > self.max_vocab_size:
            sorted_terms = sorted(kept, key=lambda t: counts[t], reverse=True)
            kept = set(sorted_terms[: self.max_vocab_size])

        # Construire le vocabulaire trié pour un ordre déterministe.
        self.vocabulary_ = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        next_id = 2  # après les tokens spéciaux
        for word in sorted(kept):
            self.vocabulary_[word] = next_id
            next_id += 1

        self.word_counts_ = counts
        return self

    # -------------------------------------------------------- _convert_one
    def _convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convertit une liste de tokens en IDs, avec troncature."""
        max_len = self.max_seq_len
        ids = [
            self.vocabulary_.get(t, UNK_ID)
            for t in tokens[:max_len]
        ]
        # Padding à droite si nécessaire.
        ids += [PAD_ID] * (max_len - len(ids))
        return ids[:max_len]

    # ----------------------------------------------------------- transform
    def transform(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Convertit une liste de textes en token IDs et masks.

        Returns:
            Tuple de deux ndarray :
            - ``input_ids``  : int64, shape ``(n_texts, max_seq_len)``
            - ``attention_mask`` : int64, shape ``(n_texts, max_seq_len)``
              (1 = token réel, 0 = padding)
        """
        all_ids = []
        all_masks = []
        for text in texts:
            tokens = tokenize_text(text)
            ids = self._convert_tokens_to_ids(tokens)
            mask = [1 if id_ != PAD_ID else 0 for id_ in ids]
            all_ids.append(ids)
            all_masks.append(mask)

        return (
            np.array(all_ids, dtype=np.int64),
            np.array(all_masks, dtype=np.int64),
        )

    def transform_one(self, text: str) -> np.ndarray:
        """Convertit un texte unique en token IDs (1D, max_seq_len).

        Retourne un ndarray int64 de shape ``(max_seq_len,)``.
        Le routeur neuronal appelle cette méthode.
        """
        tokens = tokenize_text(text)
        ids = self._convert_tokens_to_ids(tokens)
        return np.array(ids, dtype=np.int64)

    # -------------------------------------------------------------- encode
    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode un texte en (input_ids, attention_mask) — 1D chacun.

        Helper pratique pour le routeur et l'entraînement.
        """
        tokens = tokenize_text(text)
        ids = self._convert_tokens_to_ids(tokens)
        mask = [1 if id_ != PAD_ID else 0 for id_ in ids]
        return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64)

    # ---------------------------------------------------- sérialisation
    def save(self, path: str | Path) -> None:
        """Sauvegarde le tokenizer en JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "max_seq_len": self.max_seq_len,
            "vocabulary": self.vocabulary_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        """Recharge un tokenizer depuis un fichier JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(
            max_vocab_size=payload["max_vocab_size"],
            min_freq=payload["min_freq"],
            max_seq_len=payload["max_seq_len"],
        )
        tok.vocabulary_ = {k: int(v) for k, v in payload["vocabulary"].items()}
        return tok
