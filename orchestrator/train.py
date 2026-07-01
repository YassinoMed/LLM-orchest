"""Script d'entraînement du Transformer 12 couches pour le routage neuronal.

Optimisé pour NVIDIA L4 (24 Go VRAM) avec mixed precision (AMP).

Usage depuis la racine du projet :

    python -m orchestrator.train [OPTIONS]

Options principales :

    --epochs N           Nombre d'époques (défaut 30)
    --batch-size N       Taille des mini-batches (défaut 128)
    --lr RATE            Taux d'apprentissage (défaut 3e-4, adapté pour Transformer)
    --d-model DIM        Dimension du modèle (défaut 256)
    --nhead N            Têtes d'attention (défaut 8)
    --dim-ff DIM         Dimension FFN (défaut 1024)
    --num-layers N       Couches Transformer (défaut 12)
    --max-seq-len N      Longueur max des séquences (défaut 64)
    --max-vocab N        Taille du vocabulaire (défaut 10000)
    --samples-per-cat N  Exemples par catégorie (défaut 5000)
    --extra-variations N Variations enrichies par catégorie (défaut 500)
    --dropout RATE       Taux de dropout (défaut 0.1)
    --weight-decay       Régularisation L2 (défaut 1e-4)
    --output-dir         Répertoire de sortie (défaut checkpoints/)
    --data-dir           Répertoire des données (défaut data/routing_dataset)
    --seed               Graine aléatoire (défaut 42)
    --device             Device forcé (auto/cpu/mps/cuda, défaut auto)
    --no-save            Ne pas sauvegarder le modèle (pour les tests)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Configuration du logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator.train")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orchestrator.train",
        description="Entraîne le Transformer 12 couches pour le routage neuronal.",
    )
    # Modèle
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=10000)
    # Données
    parser.add_argument("--samples-per-cat", type=int, default=5000)
    parser.add_argument("--extra-variations", type=int, default=500)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--data-dir", default="data/routing_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Activer la mixed precision (défaut : True)")
    parser.add_argument("--no-amp", action="store_true", help="Désactiver AMP")
    return parser.parse_args(argv)


def _get_device(device_str: str):
    import torch
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main(argv: list[str] | None = None) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .data_generation import generate_dataset, save_dataset
    from .models import MODEL_NAMES
    from .neural_model import TransformerClassifier12
    from .tokenizer import WordTokenizer

    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _get_device(args.device)
    use_amp = args.amp and not args.no_amp and device.type == "cuda"

    logger.info("Device : %s | AMP (mixed precision) : %s", device, use_amp)
    logger.info("Catégories : %s (%d classes)", list(MODEL_NAMES), len(MODEL_NAMES))
    logger.info("Architecture : Transformer %d couches, d_model=%d, nhead=%d, ff=%d",
                 args.num_layers, args.d_model, args.nhead, args.dim_ff)

    # =====================================================================
    # 1. Génération / chargement du dataset
    # =====================================================================
    data_path = Path(args.data_dir)
    if data_path.exists():
        logger.info("Chargement du dataset existant : %s", data_path)
        from .data_generation import load_dataset
        dataset = load_dataset(data_path)
    else:
        logger.info("Génération du dataset synthétique (%d ex/cat, %d extra)...",
                     args.samples_per_cat, args.extra_variations)
        dataset = generate_dataset(
            seed=args.seed,
            samples_per_category=args.samples_per_cat,
            extra_variations=args.extra_variations,
        )
        save_dataset(dataset, data_path)

    logger.info(
        "Dataset : train=%d, val=%d, test=%d",
        len(dataset["train"]),
        len(dataset["val"]),
        len(dataset["test"]),
    )

    # =====================================================================
    # 2. Tokenisation (fit sur train uniquement)
    # =====================================================================
    logger.info("Tokenisation (max_vocab=%d, max_seq_len=%d)...",
                args.max_vocab, args.max_seq_len)
    tokenizer = WordTokenizer(
        max_vocab_size=args.max_vocab,
        min_freq=2,
        max_seq_len=args.max_seq_len,
    )

    # Fit sur le train.
    tokenizer.fit(dataset["train"]["text"])
    logger.info("Vocabulaire : %d termes (dont 2 tokens spéciaux)", tokenizer.vocab_size)

    # Tokeniser tous les splits.
    train_ids, train_masks = tokenizer.transform(dataset["train"]["text"])
    val_ids, val_masks = tokenizer.transform(dataset["val"]["text"])
    test_ids, test_masks = tokenizer.transform(dataset["test"]["text"])

    y_train = np.array(dataset["train"]["label"], dtype=np.int64)
    y_val = np.array(dataset["val"]["label"], dtype=np.int64)
    y_test = np.array(dataset["test"]["label"], dtype=np.int64)

    # Conversion en tenseurs PyTorch.
    train_ids_t = torch.from_numpy(train_ids).to(device)
    train_masks_t = torch.from_numpy(train_masks).to(device)
    val_ids_t = torch.from_numpy(val_ids).to(device)
    val_masks_t = torch.from_numpy(val_masks).to(device)
    test_ids_t = torch.from_numpy(test_ids).to(device)
    test_masks_t = torch.from_numpy(test_masks).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    # DataLoaders optimisés.
    num_workers = 4 if device.type == "cuda" else 0
    train_ds = TensorDataset(train_ids_t, train_masks_t, y_train_t)
    val_ds = TensorDataset(val_ids_t, val_masks_t, y_val_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # =====================================================================
    # 3. Modèle Transformer 12 couches
    # =====================================================================
    model = TransformerClassifier12(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(MODEL_NAMES),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

    n_params = model.count_parameters()
    logger.info("Modèle Transformer %d couches : %s paramètres",
                args.num_layers, f"{n_params:,}")

    # =====================================================================
    # 4. Entraînement avec AMP (mixed precision)
    # =====================================================================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # GradScaler pour AMP.
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = 7  # early stopping

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for ids, masks, labels in train_loader:
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(ids, masks)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids, masks)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * ids.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += ids.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for ids, masks, labels in val_loader:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(ids, masks)
                        loss = criterion(logits, labels)
                else:
                    logits = model(ids, masks)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * ids.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += ids.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
            f"{val_loss:>9.4f}  {val_acc:>7.1%}  {current_lr:>7.5f}  {elapsed:>5.1f}s"
        )

        scheduler.step()

        # Early stopping.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping à l'epoch %d (meilleur val_acc=%.1f%%)",
                             epoch, best_val_acc * 100)
                break

    # =====================================================================
    # 5. Évaluation finale sur le test set
    # =====================================================================
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                test_logits = model(test_ids_t, test_masks_t)
        else:
            test_logits = model(test_ids_t, test_masks_t)
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()

    print(f"\n{'='*70}")
    print(f"Meilleur val_acc : {best_val_acc:.1%}")
    print(f"Test accuracy    : {test_acc:.1%}")
    print(f"Nombre de paramètres : {n_params:,}")
    print(f"Device : {device} | AMP : {use_amp}")
    print(f"{'='*70}")

    # =====================================================================
    # 6. Sauvegarde
    # =====================================================================
    if not args.no_save:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        model_path = output / "model.pt"
        tokenizer_path = output / "tokenizer.json"

        # Sauvegarde modèle (sur CPU pour la portabilité).
        model.cpu()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "num_classes": len(MODEL_NAMES),
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_ff,
                "max_seq_len": args.max_seq_len,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
            },
            str(model_path),
        )

        # Sauvegarde tokenizer.
        tokenizer.save(str(tokenizer_path))

        print(f"\nCheckpoints sauvegardés :")
        print(f"  Modèle   : {model_path}")
        print(f"  Tokenizer: {tokenizer_path}")
        print(f"\nPour utiliser le mode neuronal :")
        print(f"  python cli.py 'Ta requête' --router neural")
    else:
        logger.info("Sauvegarde désactivée (--no-save)")


if __name__ == "__main__":
    main()
