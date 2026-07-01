"""Réseau de neurones : MLP à 12 couches linéaires pour la classification de texte.

Architecture (exactement 12 couches ``nn.Linear``) :

    input_dim
      → Linear(512) + ReLU + Dropout         (couche 1)
      → Linear(512) + ReLU + Dropout         (couche 2)
      → Linear(256) + ReLU + Dropout         (couche 3)
      → Linear(256) + ReLU + Dropout         (couche 4)
      → Linear(256) + ReLU + Dropout         (couche 5)
      → Linear(256) + ReLU + Dropout         (couche 6)
      → Linear(128) + ReLU + Dropout         (couche 7)
      → Linear(128) + ReLU + Dropout         (couche 8)
      → Linear(128) + ReLU + Dropout         (couche 9)
      → Linear(64)  + ReLU + Dropout         (couche 10)
      → Linear(64)  + ReLU + Dropout         (couche 11)
      → Linear(6)                           (couche 12, sortie, sans activation)

Les 6 sorties correspondent aux logits des catégories de
:data:`orchestrator.models.MODEL_NAMES`, dans l'ordre canonique.

Pourquoi 12 couches pour un classifieur de texte « simple » ? C'est un choix
déléré demandé par l'utilisateur. Le réseau est volontairement surdimensionné
pour la tâche ; l'overfitting est mitigé par le dropout, le weight decay et
l'early stopping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Dimensions des couches cachées (11 couches cachées + 1 couche de sortie = 12).
# L'entonnoir progressif (512 → 256 → 128 → 64) force une compression
# progressive de la représentation TF-IDF vers l'espace de décision à 6 classes.
HIDDEN_DIMS = (512, 512, 256, 256, 256, 256, 128, 128, 128, 64, 64)

# Nombre de catégories (fixe, correspond à MODEL_NAMES).
NUM_CLASSES = 6

# Taux de dropout par défaut — relativement élevé vu la profondeur du réseau.
DEFAULT_DROPOUT = 0.3


class MLPClassifier12(nn.Module):
    """MLP feedforward à exactement 12 couches linéaires.

    Args:
        input_dim: dimension du vecteur d'entrée (taille du vocabulaire TF-IDF).
        num_classes: nombre de classes en sortie (défaut 6 = MODEL_NAMES).
        hidden_dims: tuple des dimensions des 11 couches cachées.
        dropout: taux de dropout après chaque couche cachée (0 = pas de dropout).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        assert len(hidden_dims) == 11, (
            f"Il faut exactement 11 dimensions cachées pour 12 couches linéaires "
            f"(11 cachées + 1 sortie), reçu {len(hidden_dims)}"
        )

        # Construction séquentielle : chaque couche = Linear + ReLU + Dropout.
        layers: list[nn.Module] = []
        all_dims = [input_dim] + list(hidden_dims)

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Couche de sortie (12ème couche linéaire) : logits bruts, pas d'activation.
        layers.append(nn.Linear(all_dims[-1], num_classes))

        self.network = nn.Sequential(*layers)
        self.num_layers_linear = 12  # pour vérification externe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant.

        Args:
            x: tenseur ``(batch_size, input_dim)`` (TF-IDF).

        Returns:
            Tenseur ``(batch_size, num_classes)`` — logits non normalisés.
        """
        return self.network(x)

    # ------------------------------------------------------- utilitaires
    def count_parameters(self) -> int:
        """Nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités (softmax) au lieu des logits bruts."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# =============================================================================
# Transformer Classifier 12 couches
# =============================================================================

import math

from .tokenizer import PAD_ID


class SinusoidalPositionalEncoding(nn.Module):
    """Encodage positionnel sinusoidal (identique à "Attention Is All You Need").

    Args:
        d_model: dimension du modèle.
        max_seq_len: longueur maximale des séquences.
        dropout: taux de dropout appliqué après l'encodage.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ajoute l'encodage positionnel au tenseur d'entrée.

        Args:
            x: ``(batch, seq_len, d_model)``
        Returns:
            ``(batch, seq_len, d_model)``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier12(nn.Module):
    """Encodeur Transformer à 12 couches pour la classification de texte.

    Architecture :

        input_ids (batch, seq_len) + attention_mask (batch, seq_len)
          → Embedding + Sinusoidal Positional Encoding
          → nn.TransformerEncoder × 12 couches
          → Mean pooling masqué
          → Linear(d_model, num_classes) → logits

    Args:
        vocab_size: taille du vocabulaire (nombre de tokens).
        num_classes: nombre de classes en sortie (défaut 6 = MODEL_NAMES).
        d_model: dimension du modèle (embedding + Transformer).
        nhead: nombre de têtes d'attention.
        num_layers: nombre de couches Transformer (exactement 12).
        dim_feedforward: dimension interne des couches FFN.
        dropout: taux de dropout.
        max_seq_len: longueur maximale des séquences (pour positional encoding).
        pad_idx: ID du token de padding (les embeddings sont mis à zéro).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = NUM_CLASSES,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        pad_idx: int = PAD_ID,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.num_layers = num_layers

        # Embedding + Positional Encoding.
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Couchers du Transformer encodeur.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Couches de classification (après mean pooling).
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialisation Xavier uniforme (meilleur pour les Transformer).
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Passe avant du Transformer.

        Args:
            input_ids: tenseur int ``(batch, seq_len)`` — IDs des tokens.
            attention_mask: tenseur ``(batch, seq_len)`` — 1 = token réel,
                0 = padding. Utilisé pour masquer les positions de padding
                dans l'attention et le mean pooling.

        Returns:
            Logits ``(batch, num_classes)`` non normalisés.
        """
        # Embedding + positional encoding.
        x = self.embedding(input_ids)        # (batch, seq_len, d_model)
        x = self.pos_encoding(x)            # (batch, seq_len, d_model)

        # Masque src_key_padding pour le Transformer.
        # nn.TransformerEncoder attend True pour les positions à MASQUER.
        src_key_padding_mask = (attention_mask == 0)

        # Encoder : 12 couches.
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling masqué : ne moyenne que sur les tokens réels.
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Classification head.
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Retourne les probabilités (softmax) au lieu des logits bruts."""
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
