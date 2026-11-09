"""Train the Go policy from 100 self-play games and save the latest replay."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence, TypeAlias, TypedDict

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from model import (
    ACTION_COUNT,
    BOARD_ACTIONS,
    BOARD_SIZE,
    PASS_ACTION,
    Board,
    GoPolicyValueNet,
    Player,
    encode_observation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "python" / "checkpoint.pt"
DEFAULT_SELF_PLAY = ROOT / "python" / "latest_game.json"
MAX_GAME_MOVES = 2 * BOARD_ACTIONS
DEFAULT_GAMES_PER_ITERATION = 100
UPDATE_BATCH_SIZE = 256
TRAINING_VERSION = 3
WHITE_KOMI = 6.5
Move: TypeAlias = dict[str, int | str]


class ScoreBreakdown(TypedDict):
    territory: dict[Player, int]
    prisoners: dict[Player, int]
    komi: dict[Player, float]
    total: dict[Player, float]


@dataclass
class GameState:
    board: list[list[str | None]]
    current_player: Player
    captures: dict[str, int]
    consecutive_passes: int = 0
    ko_point: tuple[int, int] | None = None


def _neighbors(row: int, col: int) -> list[tuple[int, int]]:
    return [
        (neighbor_row, neighbor_col)
        for neighbor_row, neighbor_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        if 0 <= neighbor_row < BOARD_SIZE and 0 <= neighbor_col < BOARD_SIZE
    ]


def _group_and_liberties(
    board: Board, row: int, col: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    color = board[row][col]
    if color is None:
        return set(), set()

    stones: set[tuple[int, int]] = set()
    liberties: set[tuple[int, int]] = set()
    pending = [(row, col)]
    while pending:
        point = pending.pop()
        if point in stones:
            continue
        stones.add(point)
        for neighbor in _neighbors(*point):
            neighbor_color = board[neighbor[0]][neighbor[1]]
            if neighbor_color is None:
                liberties.add(neighbor)
            elif neighbor_color == color and neighbor not in stones:
                pending.append(neighbor)
    return stones, liberties


def _play(state: GameState, row: int, col: int) -> list[tuple[int, int]] | None:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return None
    if state.board[row][col] is not None or state.ko_point == (row, col):
        return None

    opponent: Player = "white" if state.current_player == "black" else "black"
    next_board = [line.copy() for line in state.board]
    next_board[row][col] = state.current_player
    captured: set[tuple[int, int]] = set()
    for neighbor_row, neighbor_col in _neighbors(row, col):
        if next_board[neighbor_row][neighbor_col] == opponent:
            stones, liberties = _group_and_liberties(
                next_board, neighbor_row, neighbor_col
            )
            if not liberties:
                captured.update(stones)

    for captured_row, captured_col in captured:
        next_board[captured_row][captured_col] = None

    own_stones, own_liberties = _group_and_liberties(next_board, row, col)
    if not own_liberties:
        return None

    state.board = next_board
    state.captures[state.current_player] += len(captured)
    state.consecutive_passes = 0
    state.ko_point = None
    if len(captured) == 1 and len(own_stones) == 1 and len(own_liberties) == 1:
        state.ko_point = next(iter(captured))
    state.current_player = opponent
    return sorted(captured)


def _pass(state: GameState) -> None:
    state.current_player = "white" if state.current_player == "black" else "black"
    state.consecutive_passes += 1
    state.ko_point = None


def _legal_actions(state: GameState) -> list[int]:
    actions = [PASS_ACTION]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if state.board[row][col] is not None:
                continue
            trial = GameState(
                board=state.board,
                current_player=state.current_player,
                captures=state.captures.copy(),
                ko_point=state.ko_point,
            )
            if _play(trial, row, col) is not None:
                actions.append(row * BOARD_SIZE + col)
    return actions


def _select_action(
    model: GoPolicyValueNet,
    state: GameState,
) -> tuple[int, Tensor, Tensor, Tensor, dict[str, float | int]]:
    board_tensor, captures_tensor = encode_observation(
        state.board, state.current_player, state.captures
    )
    with torch.no_grad():
        policy_logits, value = model(board_tensor, captures_tensor)
        actions = _legal_actions(state)
        legal_mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
        legal_mask[actions] = True
        legal_logits = policy_logits.masked_fill(~legal_mask.unsqueeze(0), -torch.inf)
        action_distribution = Categorical(logits=legal_logits)
        action = int(action_distribution.sample().item())
        thinking = {
            "chosen_move_probability": float(action_distribution.probs[0, action].item()),
            "value_estimate": float(value[0, 0].item()),
            "policy_entropy_bits": float(
                (action_distribution.entropy() / math.log(2)).item()
            ),
            "legal_action_count": len(actions),
        }
    return (
        action,
        board_tensor.squeeze(0),
        captures_tensor.squeeze(0),
        legal_mask,
        thinking,
    )


def _snapshot(state: GameState) -> dict[str, object]:
    return {
        "board": [line.copy() for line in state.board],
        "captures": state.captures.copy(),
        "next_player": state.current_player,
    }


def _score(state: GameState) -> tuple[Player | None, float, ScoreBreakdown]:
    """Score empty regions by their bordering color, plus prisoners and komi."""
    visited: set[tuple[int, int]] = set()
    territory = {"black": 0, "white": 0}

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if state.board[row][col] is not None or (row, col) in visited:
                continue

            region: set[tuple[int, int]] = set()
            borders: set[str] = set()
            pending = [(row, col)]
            while pending:
                point = pending.pop()
                if point in visited:
                    continue
                visited.add(point)
                region.add(point)
                for neighbor_row, neighbor_col in _neighbors(*point):
                    neighbor = state.board[neighbor_row][neighbor_col]
                    if neighbor is None:
                        if (neighbor_row, neighbor_col) not in visited:
                            pending.append((neighbor_row, neighbor_col))
                    else:
                        borders.add(neighbor)

            if len(borders) == 1:
                territory[next(iter(borders))] += len(region)

    komi = {"black": 0.0, "white": WHITE_KOMI}
    total_scores = {
        player: float(territory[player] + state.captures[player]) + komi[player]
        for player in ("black", "white")
    }
    scores: ScoreBreakdown = {
        "territory": territory,
        "prisoners": state.captures.copy(),
        "komi": komi,
        "total": total_scores,
    }
    margin = float(total_scores["black"] - total_scores["white"])
    winner = "black" if margin > 0 else "white" if margin < 0 else None
    return winner, margin, scores


def _reward_for(player: Player, winner: Player | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if player == winner else -1.0


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _play_self_play_game(
    model: GoPolicyValueNet,
    max_moves: int,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    state = GameState(
        board=[[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)],
        current_player="black",
        captures={"black": 0, "white": 0},
    )
    moves: list[dict[str, object]] = []
    positions = [_snapshot(state)]
    examples: list[dict[str, object]] = []

    for move_number in range(1, max_moves + 1):
        player = state.current_player
        action, encoded_board, encoded_captures, legal_mask, thinking = (
            _select_action(model, state)
        )
        examples.append(
            {
                "board": encoded_board,
                "captures": encoded_captures,
                "legal_mask": legal_mask,
                "action": action,
                "player": player,
            }
        )

        if action == PASS_ACTION:
            move: Move = {"type": "pass"}
            _pass(state)
        else:
            row, col = divmod(action, BOARD_SIZE)
            if _play(state, row, col) is None:
                raise RuntimeError("The policy selected a move marked as legal.")
            move = {"type": "play", "row": row, "col": col}

        moves.append(
            {
                "move_number": move_number,
                "player": player,
                "action": move,
                "captures": state.captures.copy(),
                "thinking": thinking,
            }
        )
        positions.append(_snapshot(state))
        if state.consecutive_passes >= 2:
            break

    winner, margin, scores = _score(state)
    replay: dict[str, object] = {
        "schema_version": 4,
        "board_size": BOARD_SIZE,
        "termination": (
            "two_passes" if state.consecutive_passes >= 2 else "move_limit"
        ),
        "scoring": "territory_plus_prisoners_with_6_5_komi",
        "rewards": {
            "black": _reward_for("black", winner),
            "white": _reward_for("white", winner),
        },
        "result": {"winner": winner, "margin": margin, "scores": scores},
        "moves": moves,
        "positions": positions,
    }
    buffer_game: dict[str, object] = {"winner": winner, "examples": examples}
    return replay, buffer_game, moves


def _save_torch(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary_path)
    temporary_path.replace(path)


def _update_from_games(
    model: GoPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    games: Sequence[dict[str, object]],
) -> float:
    samples = [
        (example, _reward_for(example["player"], game["winner"]))
        for game in games
        for example in game["examples"]
    ]
    if not samples:
        raise ValueError("Self-play produced no training examples.")

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for start in range(0, len(samples), UPDATE_BATCH_SIZE):
        batch = samples[start : start + UPDATE_BATCH_SIZE]
        boards = torch.stack([example["board"] for example, _ in batch])
        captures = torch.stack([example["captures"] for example, _ in batch])
        legal_masks = torch.stack([example["legal_mask"] for example, _ in batch])
        actions = torch.tensor(
            [example["action"] for example, _ in batch],
            dtype=torch.long,
        )
        rewards = torch.tensor([reward for _, reward in batch], dtype=torch.float32)

        policy_logits, values = model(boards, captures)
        policy_logits = policy_logits.masked_fill(~legal_masks, -torch.inf)
        log_probabilities = Categorical(logits=policy_logits).log_prob(actions)
        policy_loss = -(log_probabilities * rewards).mean()
        value_loss = nn.functional.mse_loss(values.squeeze(1), rewards)
        loss = policy_loss + value_loss

        (loss * (len(batch) / len(samples))).backward()
        total_loss += float(loss.detach().item()) * len(batch)

    optimizer.step()
    model.eval()
    return total_loss / len(samples)


def train_one_iteration(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    replay_path: Path = DEFAULT_SELF_PLAY,
    *,
    seed: int = 0,
    max_moves: int = MAX_GAME_MOVES,
    games_per_iteration: int = DEFAULT_GAMES_PER_ITERATION,
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    if max_moves < 2:
        raise ValueError("max_moves must be at least 2.")
    if games_per_iteration < 1:
        raise ValueError("games_per_iteration must be positive.")
    torch.manual_seed(seed)

    architecture = {"channels": 64, "residual_blocks": 3}
    iteration = 0
    checkpoint: dict[str, object] | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        iteration = int(checkpoint["iteration"])

        architecture = checkpoint["architecture"]
    model = GoPolicyValueNet(**architecture)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model checkpoint from iteration {iteration}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.eval()

    latest_replay: dict[str, object] | None = None
    latest_moves: list[dict[str, object]] = []
    new_games: list[dict[str, object]] = []
    for _ in range(games_per_iteration):
        game_replay, buffer_game, latest_moves = _play_self_play_game(
            model, max_moves
        )
        latest_replay = game_replay
        new_games.append(buffer_game)

    loss = _update_from_games(model, optimizer, new_games)

    iteration += 1
    _save_torch(
        checkpoint_path,
        {
            "training_version": TRAINING_VERSION,
            "iteration": iteration,
            "architecture": architecture,
            "model_state": model.state_dict(),
        },
    )

    if latest_replay is None:
        raise RuntimeError("No self-play games were completed.")
    latest_replay.update(
        {
            "iteration": iteration,
            "generated_at": datetime.now(UTC).isoformat(),
            "games_this_iteration": games_per_iteration,
            "training_examples": sum(len(game["examples"]) for game in new_games),
            "training_loss": loss,
        }
    )
    _write_json(replay_path, latest_replay)
    last_result = latest_replay["result"]
    return {
        "iteration": iteration,
        "games": games_per_iteration,
        "moves": len(latest_moves),
        "training_examples": sum(len(game["examples"]) for game in new_games),
        "winner": last_result["winner"],
        "margin": last_result["margin"],
        "loss": loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_SELF_PLAY)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=MAX_GAME_MOVES)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_ITERATION)
    arguments = parser.parse_args()
    seed = arguments.seed
    if seed is None:
        seed = int(datetime.now(UTC).strftime("%Y%m%d"))
    result = train_one_iteration(
        arguments.checkpoint,
        arguments.replay,
        seed=seed,
        max_moves=arguments.max_moves,
        games_per_iteration=arguments.games,
    )
    print(
        "Completed iteration {iteration}: {games} games, {training_examples} "
        "training positions, latest game winner={winner}, margin={margin:g}, "
        "loss={loss:.4f}".format(**result)
    )


if __name__ == "__main__":
    main()
