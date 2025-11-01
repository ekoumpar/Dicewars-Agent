# RL Agent for Dicewars

This repository implements an **AI agent** for the strategy game *Dicewars*, trained using **Reinforcement Learning (RL)** techniques in **Python**.

---

## Overview

This project was developed as part of the *Machine Learning* course at the **University of Twente**.

The main objective is to create an **autonomous player** that learns to play *Dicewars* by interacting with the environment using reinforcement learning principles.

The base game environment, framework, and helper scripts were provided by the course instructors.  
Our implementation was evaluated in a **student tournament** where agents competed based on their learned strategies.

A detailed explanation of the approach, experiments, and results is available in the [Report](https://github.com/ekoumpar/Dicewars-Agent/report/report.pdf).

---

## Implementation Details

The agent is based on the **Proximal Policy Optimization (PPO)** algorithm — a policy-gradient method known for its **stability**, **robustness**, and **sample efficiency**.

The system is composed of two main components:

- **Player Module**  
  Handles the game logic, state encoding, and action selection.  
  Interfaces directly with the provided game environment.

- **PPO Agent Module**  
  Implements the reinforcement learning algorithm using an **actor–critic** architecture.  
  Responsible for policy updates, reward processing, and training loop management.

When pretrained model weights (`actor.weights.h5` and `critic.weights.h5`) are available in the project directory, the agent automatically **loads the saved weights** to play immediately.  
If no weights are found, the model will **train from scratch**.

Training progress is visualized through **loss** and **reward** plots for tuning and performance evaluation.

---







