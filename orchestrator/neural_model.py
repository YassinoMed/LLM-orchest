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
