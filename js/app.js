(function () {
  "use strict";

  const { createGame } = window.GoEngine;
  const boardElement = document.querySelector("#board");
  const blackCaptures = document.querySelector("#black-captures");
  const whiteCaptures = document.querySelector("#white-captures");
  const statusElement = document.querySelector("#game-status");
  const iterationLabel = document.querySelector("#iteration-label");
  const thinkingProbability = document.querySelector("#thinking-probability");
  const thinkingValue = document.querySelector("#thinking-value");
  const thinkingEntropy = document.querySelector("#thinking-entropy");
  const thinkingLegalActions = document.querySelector("#thinking-legal-actions");
  const previousButton = document.querySelector("#previous-button");
  const playButton = document.querySelector("#play-button");
  const nextButton = document.querySelector("#next-button");
  const replaySlider = document.querySelector("#replay-slider");

  let game = createGame(9);
  const intersections = [];
  let replay = null;
  let positionIndex = 0;
  let playbackTimer = null;

  function buildBoard() {
    boardElement.replaceChildren();
    intersections.length = 0;

    for (let row = 0; row < game.size; row += 1) {
      for (let col = 0; col < game.size; col += 1) {
        const intersection = document.createElement("span");
        intersection.className = "intersection";
        intersection.setAttribute("aria-hidden", "true");
        boardElement.append(intersection);
        intersections.push(intersection);
      }
    }
  }

  function renderPosition() {
    const position = replay.positions[positionIndex];
    game.board = position.board;
    game.captures = position.captures;
    blackCaptures.textContent = `Black captures: ${game.captures.black}`;
    whiteCaptures.textContent = `White captures: ${game.captures.white}`;
    iterationLabel.textContent = `Iteration ${replay.iteration}`;
    const move = positionIndex > 0 ? replay.moves[positionIndex - 1] : null;
    if (move) {
      const { thinking } = move;
      thinkingProbability.textContent =
        `${(thinking.chosen_move_probability * 100).toFixed(1)}%`;
      thinkingValue.textContent =
        `${thinking.value_estimate >= 0 ? "+" : ""}${thinking.value_estimate.toFixed(2)}`;
      thinkingEntropy.textContent = `${thinking.policy_entropy_bits.toFixed(2)} bits`;
      thinkingLegalActions.textContent = String(thinking.legal_action_count);
    } else {
      thinkingProbability.textContent = "—";
      thinkingValue.textContent = "—";
      thinkingEntropy.textContent = "—";
      thinkingLegalActions.textContent = "—";
    }
    replaySlider.value = String(positionIndex);
    previousButton.disabled = positionIndex === 0;
    nextButton.disabled = positionIndex >= replay.moves.length;
    playButton.disabled = replay.moves.length === 0;
    boardElement.setAttribute("aria-label", "9 by 9 Go board at selected replay position");

    intersections.forEach((intersection, index) => {
      const row = Math.floor(index / game.size);
      const col = index % game.size;
      const stone = game.board[row][col];
      const isLastMoveStone =
        move?.action.type === "play" &&
        move.action.row === row &&
        move.action.col === col &&
        stone === move.player;
      if (stone) {
        intersection.dataset.player = stone;
      } else {
        delete intersection.dataset.player;
      }
      if (isLastMoveStone) {
        intersection.dataset.lastMove = "true";
      } else {
        delete intersection.dataset.lastMove;
      }
    });
  }

  function stopPlayback() {
    if (playbackTimer !== null) {
      window.clearInterval(playbackTimer);
      playbackTimer = null;
    }
    playButton.textContent = "Play";
    playButton.setAttribute("aria-pressed", "false");
  }

  function advanceReplay() {
    if (positionIndex >= replay.moves.length) {
      stopPlayback();
      return;
    }
    positionIndex += 1;
    renderPosition();
  }

  function showReplayError(error) {
    console.error("Could not load the Go replay.", error);
    statusElement.textContent = "The self-play replay could not be loaded.";
    iterationLabel.textContent = "";
    previousButton.disabled = true;
    playButton.disabled = true;
    nextButton.disabled = true;
    replaySlider.disabled = true;
  }

  async function loadReplay() {
    try {
      const response = await fetch("game.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Replay request failed with status ${response.status}.`);
      }

      const data = await response.json();
      if (
        data.schema_version !== 4 ||
        data.board_size !== game.size ||
        !Array.isArray(data.moves) ||
        !Array.isArray(data.positions) ||
        data.positions.length !== data.moves.length + 1
      ) {
        throw new Error("Replay data has an unsupported format.");
      }
      replay = data;
      replaySlider.max = String(replay.moves.length);
      replaySlider.disabled = false;
      statusElement.textContent = "";
      renderPosition();
    } catch (error) {
      showReplayError(error);
    }
  }

  previousButton.addEventListener("click", () => {
    if (positionIndex > 0) {
      stopPlayback();
      positionIndex -= 1;
      renderPosition();
    }
  });
  nextButton.addEventListener("click", () => {
    stopPlayback();
    advanceReplay();
  });
  replaySlider.addEventListener("input", () => {
    stopPlayback();
    positionIndex = Number(replaySlider.value);
    renderPosition();
  });
  playButton.addEventListener("click", () => {
    if (playbackTimer !== null) {
      stopPlayback();
      return;
    }
    if (positionIndex >= replay.moves.length) {
      positionIndex = 0;
      renderPosition();
    }
    playButton.textContent = "Pause";
    playButton.setAttribute("aria-pressed", "true");
    playbackTimer = window.setInterval(advanceReplay, 700);
  });

  buildBoard();
  statusElement.textContent = "Loading the latest self-play replay…";
  loadReplay();
})();
