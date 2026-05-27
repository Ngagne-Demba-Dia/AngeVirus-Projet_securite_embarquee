"""
models.py — Architectures partagées entre 03_train.py, 04_transfer.py, 05_monitor.py et api/serve.py
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    CNN 1D opérant sur le vecteur de features (325 dims = 25 fenêtres × 13 features).
    3 blocs Conv1d conformément au cahier des charges (PDF étape 3).
    """
    def __init__(self, n_input: int = 325, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            # Bloc 1
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            # Bloc 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            # Bloc 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_input) → (B, 1, n_input) pour Conv1d
        return self.classifier(self.conv(x.unsqueeze(1)))
