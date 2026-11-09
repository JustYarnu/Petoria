import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor, nn

import train_one_iteration as trainer


class TinyPolicyValueNet(nn.Module):
    def __init__(self, channels: int = 1, residual_blocks: int = 0) -> None:
        super().__init__()
        del channels, residual_blocks
        self.policy = nn.Parameter(torch.zeros(trainer.ACTION_COUNT))
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, board: Tensor, captures: Tensor) -> tuple[Tensor, Tensor]:
        del captures
        return (
            self.policy.expand(board.shape[0], -1),
            self.value.expand(board.shape[0], 1).tanh(),
        )


def example() -> dict[str, object]:
    legal_mask = torch.zeros(trainer.ACTION_COUNT, dtype=torch.bool)
    legal_mask[:2] = True
    return {
        "board": torch.zeros(3, trainer.BOARD_SIZE, trainer.BOARD_SIZE),
        "captures": torch.zeros(2),
        "legal_mask": legal_mask,
        "action": 0,
        "player": "black",
    }


class TrainOneIterationTests(unittest.TestCase):
    def test_default_iteration_plays_100_games(self) -> None:
        self.assertEqual(trainer.DEFAULT_GAMES_PER_ITERATION, 100)

    def test_rewards_are_terminal_win_and_loss(self) -> None:
        self.assertEqual(trainer._reward_for("black", "black"), 1.0)
        self.assertEqual(trainer._reward_for("white", "black"), -1.0)
        self.assertEqual(trainer._reward_for("black", None), 0.0)

    def test_self_play_uses_one_model_for_both_players(self) -> None:
        model = TinyPolicyValueNet()
        legal_mask = torch.zeros(trainer.ACTION_COUNT, dtype=torch.bool)
        legal_mask[trainer.PASS_ACTION] = True
        selected_pass = (
            trainer.PASS_ACTION,
            torch.zeros(3, trainer.BOARD_SIZE, trainer.BOARD_SIZE),
            torch.zeros(2),
            legal_mask,
            {
                "chosen_move_probability": 1.0,
                "value_estimate": 0.0,
                "policy_entropy_bits": 0.0,
                "legal_action_count": 1,
            },
        )
        with patch.object(
            trainer, "_select_action", return_value=selected_pass
        ) as select_action:
            replay, _, _ = trainer._play_self_play_game(model, max_moves=4)

        self.assertEqual(select_action.call_count, 2)
        self.assertTrue(
            all(call.args[0] is model for call in select_action.call_args_list)
        )
        self.assertEqual(replay["rewards"], {"black": -1.0, "white": 1.0})

    def test_update_uses_rewards_to_change_policy_and_value(self) -> None:
        model = TinyPolicyValueNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        trainer._update_from_games(
            model,
            optimizer,
            [{"winner": "black", "examples": [example()]}],
        )

        self.assertGreater(model.policy[0].item(), model.policy[1].item())
        self.assertGreater(model.value.item(), 0.0)

    def test_iteration_loads_checkpoint_runs_games_and_saves_updated_model(self) -> None:
        architecture = {"channels": 1, "residual_blocks": 0}
        initial_model = TinyPolicyValueNet(**architecture)
        initial_state = {
            key: value.clone() for key, value in initial_model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            replay_path = Path(directory) / "replay.json"
            torch.save(
                {
                    "iteration": 12,
                    "architecture": architecture,
                    "model_state": initial_state,
                },
                checkpoint_path,
            )
            game = (
                {"result": {"winner": "black", "margin": 1.0}},
                {"winner": "black", "examples": [example()]},
                [],
            )
            with (
                patch.object(trainer, "GoPolicyValueNet", TinyPolicyValueNet),
                patch.object(
                    trainer, "_play_self_play_game", return_value=game
                ) as play_game,
            ):
                result = trainer.train_one_iteration(
                    checkpoint_path,
                    replay_path,
                    games_per_iteration=2,
                    max_moves=2,
                )

            saved_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(result["iteration"], 13)
            self.assertEqual(result["games"], 2)
            self.assertEqual(play_game.call_count, 2)
            self.assertFalse(
                torch.equal(
                    saved_checkpoint["model_state"]["policy"],
                    initial_state["policy"],
                )
            )
            self.assertTrue(replay_path.is_file())


if __name__ == "__main__":
    unittest.main()
