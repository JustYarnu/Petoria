"""PyTorch state encoder and policy/value network for 9x9 Go self-play."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn

BOARD_SIZE = 9
BOARD_CHANNELS = 3
CAPTURE_FEATURES = 2
BOARD_ACTIONS = BOARD_SIZE * BOARD_SIZE
PASS_ACTION = BOARD_ACTIONS
ACTION_COUNT = BOARD_ACTIONS + 1

Player: TypeAlias = Literal["black", "white"]
Stone: TypeAlias = Player | None
Board: TypeAlias = Sequence[Sequence[Stone]]


def encode_observation(
    board: Board,
    current_player: Player,
    captures: Mapping[Player, int],
) -> tuple[Tensor, Tensor]:
    """Encode a 9x9 board and captures into network-ready, batched tensors.

    Board planes are current-player stones, opponent stones, and a constant
    plane that is 1 when Black is to move (otherwise 0). Capture features are
    the current player's and opponent's counts, in that order and log-scaled
    relative to the 81-point board size.
    """
    if current_player not in ("black", "white"):
        raise ValueError("current_player must be 'black' or 'white'.")
    if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
        raise ValueError(f"board must have shape ({BOARD_SIZE}, {BOARD_SIZE}).")

    board_tensor = torch.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    opponent = "white" if current_player == "black" else "black"
    for row_index, row in enumerate(board):
        for col_index, stone in enumerate(row):
            if stone is None:
                continue
            if stone == current_player:
                board_tensor[0, row_index, col_index] = 1
            elif stone == opponent:
                board_tensor[1, row_index, col_index] = 1
            else:
                raise ValueError(
                    f"invalid stone at ({row_index}, {col_index}): {stone!r}"
                )

    board_tensor[2].fill_(1 if current_player == "black" else 0)
    capture_values = []
    for player in (current_player, opponent):
        count = captures.get(player)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"captures[{player!r}] must be a non-negative integer.")
        capture_values.append(math.log1p(count) / math.log1p(BOARD_ACTIONS))

    return (
        board_tensor.unsqueeze(0),
        torch.tensor([capture_values], dtype=torch.float32),
    )


class ResidualBlock(nn.Module):
    """A two-convolution residual block for spatial board features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.layers(inputs))


class GoPolicyValueNet(nn.Module):
    """Predict a move policy and current-player outcome value from a Go state.

    Policy indices 0-80 correspond to board points in row-major order; index
    81 is pass. The value is an expected outcome in [-1, 1] for the player to
    move. Capture inputs are ordered Black, then White.
    """

    def __init__(self, channels: int = 64, residual_blocks: int = 3) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        if residual_blocks < 0:
            raise ValueError("residual_blocks cannot be negative.")

        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *(ResidualBlock(channels) for _ in range(residual_blocks)),
        )
        self.capture_projection = nn.Linear(CAPTURE_FEATURES, channels)
        self.board_policy = nn.Conv2d(channels, 1, kernel_size=1)
        conditioned_features = channels + CAPTURE_FEATURES
        self.pass_policy = nn.Linear(conditioned_features, 1)
        self.value_head = nn.Sequential(
            nn.Linear(conditioned_features, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, board: Tensor, captures: Tensor) -> tuple[Tensor, Tensor]:
        if board.ndim != 4 or board.shape[1:] != (
            BOARD_CHANNELS,
            BOARD_SIZE,
            BOARD_SIZE,
        ):
            raise ValueError(
                f"board must have shape (batch, {BOARD_CHANNELS}, "
                f"{BOARD_SIZE}, {BOARD_SIZE})."
            )
        if captures.shape != (board.shape[0], CAPTURE_FEATURES):
            raise ValueError(
                f"captures must have shape (batch, {CAPTURE_FEATURES})."
            )

        features = self.trunk(board)
        pooled = features.mean(dim=(2, 3))
        conditioned = torch.cat((pooled, captures), dim=1)
        capture_context = self.capture_projection(captures).unsqueeze(-1).unsqueeze(
            -1
        )
        board_logits = self.board_policy(
            features + capture_context
        ).flatten(start_dim=1)
        pass_logit = self.pass_policy(conditioned)
        policy_logits = torch.cat((board_logits, pass_logit), dim=1)
        value = self.value_head(conditioned)
        return policy_logits, value
