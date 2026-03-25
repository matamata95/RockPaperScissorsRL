# RockPaperScissorsRL

A small reinforcement-learning playground for Rock–Paper–Scissors (RPS).

This repository implements a minimal RPS environment, several baseline agents, training/evaluation scripts, and utilities for explainability (SHAP).

## Features

- Lightweight custom environment: `src/RPS_env.py` — supports best-of-N rounds and a rolling history buffer.
- Baseline agents: Q-learning, random, and win-stay in `src/agents/`.
- Training scripts: `src/train.py`, `src/train_vs_agent.py` for self-play and opponent training.
- Play against script: `src/play.py`.
- Explainability helpers and saved dataset (dataset is derived from states generated during training): `explainable_shap_models/`.

## Quick setup

Recommended: create the conda environment in `requirements/environment.yml`:

## Project layout

- `src/`
	- `RPS_env.py` — environment implementation
	- `train.py` — training entrypoint
	- `train_vs_agent.py` — training vs a fixed agent
	- `play.py` — run/play episodes
	- `explain_shap.py` — SHAP explainability
	- `agents/` — baseline agents (`q_learning_agent.py`, `random_agent.py`, `win_stay_agent.py`, `play_previous_agent.py`)
- `explainable_shap_models/` — dataset and explainer outputs used for model explanation
- `requirements/` — environment spec

## Notes

- The environment uses a numeric action space [0,1,2] mapping to [rock, paper, scissors].
- History of previous states is implemented as a rolling buffer with deque and can be changed in 'src/train.py'.
- Rewards depending on the Best-Of-Games played, where a win/loss in a round is +1/-1, where as the win/loss of a game is +5/-5.
- Set random seeds in the training scripts for reproducibility.
- Explainability is achieved with SHAP by training a decision tree regression model for each action. Features are derived from history buffer and formatted in the following way: 'h{i}_p{j}_R/P/S'. h{i} corresponds to a place in the history buffer (how long ago this hand was played) where the lower numbers are oldest actions, since it was implemented with deque: 
    -buffer_size == 2:
        - h0_p1_rock = player 1 played a rock 2 rounds ago | h ∈ [buffer_size]
        - h1_p2_paper = player 2 played paper last round | p ∈ [1, 2]
- This is a personal project to get accustomed to reinforcement-learning and explainability implementations.

