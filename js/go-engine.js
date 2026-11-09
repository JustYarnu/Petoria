(function (root, factory) {
  const engine = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = engine;
  } else {
    root.GoEngine = engine;
  }
})(typeof globalThis === "undefined" ? this : globalThis, function () {
  "use strict";

  const BLACK = "black";
  const WHITE = "white";

  function createGame(size = 9) {
    if (!Number.isInteger(size) || size < 2 || size > 25) {
      throw new RangeError("Board size must be an integer from 2 to 25.");
    }

    return {
      size,
      board: Array.from({ length: size }, () => Array(size).fill(null)),
      currentPlayer: BLACK,
      captures: { black: 0, white: 0 },
      moveNumber: 0,
      koPoint: null,
      consecutivePasses: 0,
      gameOver: false,
    };
  }

  function otherPlayer(player) {
    return player === BLACK ? WHITE : BLACK;
  }

  function getNeighbors(size, row, col) {
    return [
      [row - 1, col],
      [row + 1, col],
      [row, col - 1],
      [row, col + 1],
    ].filter(([neighborRow, neighborCol]) =>
      neighborRow >= 0 &&
      neighborRow < size &&
      neighborCol >= 0 &&
      neighborCol < size
    );
  }

  function collectGroup(board, row, col) {
    const color = board[row][col];
    const size = board.length;
    const stones = [];
    const liberties = new Set();
    const visited = new Set();
    const pending = [[row, col]];

    while (pending.length > 0) {
      const [currentRow, currentCol] = pending.pop();
      const key = `${currentRow},${currentCol}`;

      if (visited.has(key)) {
        continue;
      }

      visited.add(key);
      stones.push([currentRow, currentCol]);

      for (const [neighborRow, neighborCol] of getNeighbors(size, currentRow, currentCol)) {
        const neighbor = board[neighborRow][neighborCol];
        if (neighbor === null) {
          liberties.add(`${neighborRow},${neighborCol}`);
        } else if (neighbor === color) {
          pending.push([neighborRow, neighborCol]);
        }
      }
    }

    return { stones, liberties };
  }

  function evaluateMove(game, row, col) {
    if (!Number.isInteger(row) || !Number.isInteger(col) ||
        row < 0 || row >= game.size || col < 0 || col >= game.size) {
      return { valid: false, reason: "out_of_bounds" };
    }
    if (game.gameOver) {
      return { valid: false, reason: "game_over" };
    }
    if (game.board[row][col] !== null) {
      return { valid: false, reason: "occupied" };
    }
    if (game.koPoint && game.koPoint.row === row && game.koPoint.col === col) {
      return { valid: false, reason: "ko" };
    }

    const player = game.currentPlayer;
    const opponent = otherPlayer(player);
    const board = game.board.map((line) => line.slice());
    const captured = [];
    board[row][col] = player;

    for (const [neighborRow, neighborCol] of getNeighbors(game.size, row, col)) {
      if (board[neighborRow][neighborCol] !== opponent) {
        continue;
      }

      const group = collectGroup(board, neighborRow, neighborCol);
      if (group.liberties.size === 0) {
        for (const [stoneRow, stoneCol] of group.stones) {
          board[stoneRow][stoneCol] = null;
          captured.push({ row: stoneRow, col: stoneCol });
        }
      }
    }

    const ownGroup = collectGroup(board, row, col);
    if (ownGroup.liberties.size === 0) {
      return { valid: false, reason: "suicide" };
    }

    return { valid: true, player, board, captured, ownGroup };
  }

  function validateMove(game, row, col) {
    const move = evaluateMove(game, row, col);
    if (!move.valid) {
      return { valid: false, reason: move.reason };
    }
    return {
      valid: true,
      player: move.player,
      captured: move.captured,
    };
  }

  function placeStone(game, row, col) {
    const move = evaluateMove(game, row, col);
    if (!move.valid) {
      return { valid: false, reason: move.reason };
    }

    game.board = move.board;
    game.captures[move.player] += move.captured.length;
    game.moveNumber += 1;
    game.consecutivePasses = 0;
    game.koPoint = null;

    if (move.captured.length === 1 &&
        move.ownGroup.stones.length === 1 &&
        move.ownGroup.liberties.size === 1) {
      game.koPoint = move.captured[0];
    }

    return {
      valid: true,
      player: move.player,
      captured: move.captured,
    };
  }

  function takeTurn(game, row, col) {
    const result = placeStone(game, row, col);
    if (result.valid) {
      game.currentPlayer = otherPlayer(game.currentPlayer);
    }
    return result;
  }

  function passTurn(game) {
    if (game.gameOver) {
      return { valid: false, reason: "game_over" };
    }

    const player = game.currentPlayer;
    game.currentPlayer = otherPlayer(player);
    game.moveNumber += 1;
    game.consecutivePasses += 1;
    game.koPoint = null;
    game.gameOver = game.consecutivePasses >= 2;

    return { valid: true, player, gameOver: game.gameOver };
  }

  return {
    BLACK,
    WHITE,
    createGame,
    validateMove,
    placeStone,
    takeTurn,
    passTurn,
  };
});
