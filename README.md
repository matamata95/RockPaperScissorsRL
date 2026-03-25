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

- The environment uses a numeric action space [0,1,2] mapping to [rock, paper, scissors]. See `src/RPS_env.py` for reward shaping and history buffer details.
- Set random seeds in the training scripts for reproducibility.
- This is a personal project to get accustomed to reinforcement-learning and explainability implementations.

