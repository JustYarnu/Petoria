"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { createGame, placeStone, takeTurn, passTurn, validateMove } = require("./go-engine");

test("creates an empty 9x9 board and alternates turns", () => {
  const game = createGame();

  assert.equal(game.board.length, 9);
  assert.equal(game.board.every((row) => row.length === 9 && row.every((point) => point === null)), true);
  assert.equal(game.currentPlayer, "black");
  assert.equal(takeTurn(game, 4, 4).valid, true);
  assert.equal(game.currentPlayer, "white");
  assert.equal(takeTurn(game, 4, 4).reason, "occupied");
  assert.equal(game.currentPlayer, "white");
});

test("places a stone without advancing the current player", () => {
  const game = createGame();

  assert.deepEqual(placeStone(game, 4, 4), {
    valid: true,
    player: "black",
    captured: [],
  });
  assert.equal(game.board[4][4], "black");
  assert.equal(game.currentPlayer, "black");
});

test("captures a surrounded stone and counts it for the capturing player", () => {
  const game = createGame();
  const moves = [
    [8, 8], [1, 1], [0, 1], [7, 8], [1, 0],
    [7, 7], [2, 1], [7, 6],
  ];

  for (const [row, col] of moves) {
    assert.equal(takeTurn(game, row, col).valid, true);
  }

  const capture = takeTurn(game, 1, 2);
  assert.equal(capture.valid, true);
  assert.equal(capture.captured.length, 1);
  assert.equal(game.board[1][1], null);
  assert.equal(game.captures.black, 1);
});

test("captures separate opponent groups sharing their last liberty", () => {
  const game = createGame(5);
  game.board[0][1] = "white";
  game.board[1][0] = "white";
  game.board[0][2] = "black";
  game.board[1][1] = "black";
  game.board[2][0] = "black";

  const result = takeTurn(game, 0, 0);

  assert.equal(result.valid, true);
  assert.equal(result.captured.length, 2);
  assert.equal(game.board[0][1], null);
  assert.equal(game.board[1][0], null);
  assert.equal(game.captures.black, 2);
});

test("rejects suicide without changing the board or turn", () => {
  const game = createGame();
  game.currentPlayer = "black";
  game.board[0][1] = "white";
  game.board[1][0] = "white";
  game.board[1][2] = "white";
  game.board[2][1] = "white";

  assert.deepEqual(validateMove(game, 1, 1), { valid: false, reason: "suicide" });
  assert.equal(takeTurn(game, 1, 1).reason, "suicide");
  assert.equal(game.board[1][1], null);
  assert.equal(game.currentPlayer, "black");
});

test("rejects an immediate ko recapture", () => {
  const game = createGame();
  game.currentPlayer = "white";
  game.board[1][1] = "black";
  game.board[0][1] = "white";
  game.board[1][0] = "white";
  game.board[2][1] = "white";
  game.board[0][2] = "black";
  game.board[2][2] = "black";
  game.board[1][3] = "black";

  assert.equal(takeTurn(game, 1, 2).valid, true);
  assert.deepEqual(game.koPoint, { row: 1, col: 1 });
  assert.deepEqual(validateMove(game, 1, 1), { valid: false, reason: "ko" });
});

test("two consecutive passes end the game", () => {
  const game = createGame();

  assert.deepEqual(passTurn(game), { valid: true, player: "black", gameOver: false });
  assert.deepEqual(passTurn(game), { valid: true, player: "white", gameOver: true });
  assert.equal(takeTurn(game, 4, 4).reason, "game_over");
  assert.equal(passTurn(game).reason, "game_over");
});
