"""Validate the latest self-play replay and publish it for the static site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "python" / "latest_game.json"
DEFAULT_DESTINATION = ROOT / "game.json"
BOARD_SIZE = 9


def validate_replay(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Replay must be a JSON object.")
    if data.get("schema_version") != 4 or data.get("board_size") != BOARD_SIZE:
        raise ValueError("Replay has an unsupported schema or board size.")
    if data.get("termination") not in ("two_passes", "move_limit"):
        raise ValueError("Replay must identify why self-play ended.")
    positions = data.get("positions")
    moves = data.get("moves")
    if not isinstance(positions, list) or not isinstance(moves, list):
        raise ValueError("Replay must contain positions and moves arrays.")
    if len(positions) != len(moves) + 1:
        raise ValueError("Replay must contain an initial position and one per move.")

    for position in positions:
        if not isinstance(position, dict):
            raise ValueError("Each position must be an object.")
        board = position.get("board")
        captures = position.get("captures")
        if (
            not isinstance(board, list)
            or len(board) != BOARD_SIZE
            or any(not isinstance(row, list) or len(row) != BOARD_SIZE for row in board)
            or any(stone not in (None, "black", "white") for row in board for stone in row)
        ):
            raise ValueError("Every replay board must be a 9x9 array of stones.")
        if (
            not isinstance(captures, dict)
            or any(
                isinstance(captures.get(player), bool)
                or not isinstance(captures.get(player), int)
                or captures[player] < 0
                for player in ("black", "white")
            )
        ):
            raise ValueError("Every position must have non-negative capture counts.")

    for index, move in enumerate(moves, start=1):
        if not isinstance(move, dict) or move.get("move_number") != index:
            raise ValueError("Moves must be numbered consecutively from one.")
        if move.get("player") not in ("black", "white"):
            raise ValueError("Every move must identify its player.")
        action = move.get("action")
        if not isinstance(action, dict):
            raise ValueError("Every move must have an action object.")
        if action.get("type") == "pass":
            pass
        elif (
            action.get("type") != "play"
            or isinstance(action.get("row"), bool)
            or not isinstance(action.get("row"), int)
            or isinstance(action.get("col"), bool)
            or not isinstance(action.get("col"), int)
            or not 0 <= action["row"] < BOARD_SIZE
            or not 0 <= action["col"] < BOARD_SIZE
        ):
            raise ValueError("Move actions must be pass or a valid board coordinate.")
        thinking = move.get("thinking")
        if not isinstance(thinking, dict):
            raise ValueError("Every move must include model thinking statistics.")
        probability = thinking.get("chosen_move_probability")
        value_estimate = thinking.get("value_estimate")
        entropy = thinking.get("policy_entropy_bits")
        legal_action_count = thinking.get("legal_action_count")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
            or isinstance(value_estimate, bool)
            or not isinstance(value_estimate, (int, float))
            or not -1 <= value_estimate <= 1
            or isinstance(entropy, bool)
            or not isinstance(entropy, (int, float))
            or entropy < 0
            or isinstance(legal_action_count, bool)
            or not isinstance(legal_action_count, int)
            or not 1 <= legal_action_count <= BOARD_SIZE * BOARD_SIZE + 1
        ):
            raise ValueError("Model thinking statistics have invalid values.")


def update_frontend(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Self-play replay does not exist: {source}")
    replay = json.loads(source.read_text(encoding="utf-8"))
    validate_replay(replay)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
    temporary_destination.write_text(
        json.dumps(replay, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_destination.replace(destination)
    print(f"Published iteration {replay['iteration']} replay to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    arguments = parser.parse_args()
    update_frontend(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
