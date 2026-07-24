from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_hidden_layers = max(1, int(num_hidden_layers))
        self.dropout = float(dropout)

        layers: list[nn.Module] = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))
        for _ in range(self.num_hidden_layers - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FEMNISTCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        image_size: tuple[int, int] = (28, 28),
        input_preprocessing: str = "none",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.image_size = tuple(int(v) for v in image_size)
        self.input_preprocessing = str(input_preprocessing)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        conv_output_dim = self._infer_conv_output_dim()
        self.fc1 = nn.Sequential(
            nn.Linear(conv_output_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def _infer_conv_output_dim(self) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, 1, *self.image_size)
            sample = self.conv2(self.conv1(sample))
            return int(sample.numel())

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            expected_dim = self.image_size[0] * self.image_size[1]
            if int(x.shape[1]) != expected_dim:
                raise ValueError(f"Expected flattened FEMNIST inputs with dim={expected_dim}, got {tuple(x.shape)}")
            return x.reshape(x.shape[0], 1, *self.image_size)
        if x.ndim == 3:
            return x.unsqueeze(1)
        if x.ndim == 4:
            return x
        raise ValueError(f"Unsupported FEMNIST tensor shape: {tuple(x.shape)}")

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_preprocessing == "none":
            return x
        if self.input_preprocessing == "samplewise_standardize":
            # FEMNIST l2 inputs are nearly constant because white backgrounds dominate each row.
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
            return (x - mean) / std
        raise ValueError(f"Unsupported FEMNIST input preprocessing: {self.input_preprocessing}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        x = self._preprocess_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout_layer(x)
        return self.fc(x)


class DFLNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        branch_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.branch_layers = max(1, int(branch_layers))
        self.dropout = float(dropout)

        self.global_branch = self._make_branch(self.input_dim, self.hidden_dim, self.branch_layers, self.dropout)
        self.local_branch = self._make_branch(self.input_dim, self.hidden_dim, self.branch_layers, self.dropout)
        self.head = nn.Linear(self.hidden_dim * 2, self.num_classes)

    @staticmethod
    def _make_branch(input_dim: int, hidden_dim: int, branch_layers: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(branch_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        invariant = self.global_branch(x)
        specific = self.local_branch(x)
        logits = self.head(torch.cat([invariant, specific], dim=1))
        return logits, invariant, specific

    def forward_global_branch(self, x: torch.Tensor, *, branch: nn.Module | None = None) -> torch.Tensor:
        active_branch = self.global_branch if branch is None else branch
        return active_branch(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_branches(x)
        return logits


class _DFLFEMNISTCNNBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        image_size: tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.image_size = tuple(int(v) for v in image_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        conv_output_dim = self._infer_conv_output_dim()
        self.fc1 = nn.Sequential(
            nn.Linear(conv_output_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

    def _infer_conv_output_dim(self) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, 1, *self.image_size)
            sample = self.conv2(self.conv1(sample))
            return int(sample.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return self.dropout_layer(x)


class DFLFEMNISTCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        image_size: tuple[int, int] = (28, 28),
        input_preprocessing: str = "none",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.image_size = tuple(int(v) for v in image_size)
        self.input_preprocessing = str(input_preprocessing)

        self.global_branch = _DFLFEMNISTCNNBranch(
            self.hidden_dim,
            dropout=self.dropout,
            image_size=self.image_size,
        )
        self.local_branch = _DFLFEMNISTCNNBranch(
            self.hidden_dim,
            dropout=self.dropout,
            image_size=self.image_size,
        )
        self.head = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            expected_dim = self.image_size[0] * self.image_size[1]
            if int(x.shape[1]) != expected_dim:
                raise ValueError(f"Expected flattened FEMNIST inputs with dim={expected_dim}, got {tuple(x.shape)}")
            return x.reshape(x.shape[0], 1, *self.image_size)
        if x.ndim == 3:
            return x.unsqueeze(1)
        if x.ndim == 4:
            return x
        raise ValueError(f"Unsupported FEMNIST tensor shape: {tuple(x.shape)}")

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_preprocessing == "none":
            return x
        if self.input_preprocessing == "samplewise_standardize":
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
            return (x - mean) / std
        raise ValueError(f"Unsupported FEMNIST input preprocessing: {self.input_preprocessing}")

    def forward_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._reshape_input(x)
        x = self._preprocess_input(x)
        invariant = self.global_branch(x)
        specific = self.local_branch(x)
        logits = self.head(torch.cat([invariant, specific], dim=1))
        return logits, invariant, specific

    def forward_global_branch(self, x: torch.Tensor, *, branch: nn.Module | None = None) -> torch.Tensor:
        x = self._reshape_input(x)
        x = self._preprocess_input(x)
        active_branch = self.global_branch if branch is None else branch
        return active_branch(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_branches(x)
        return logits
