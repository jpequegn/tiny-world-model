# Tiny World Model

A minimal world model trained on [Gymnasium](https://gymnasium.farama.org/) environments.

## Motivation

World models learn a compressed representation of environment dynamics — given a current state and action, they predict the next state and reward. This enables planning without interacting with the real environment, drastically improving sample efficiency.

This project implements a compact world model to explore:
- Latent state representation via an encoder/decoder
- Transition dynamics in latent space
- Model-based planning and policy learning

## Architecture

```
Observation (s_t)  →  Encoder  →  Latent (z_t)
                                       │
                           Action (a_t)┤
                                       ↓
                               Transition Model
                                       │
                                       ↓
                              Latent (z_{t+1})  →  Decoder  →  Predicted Observation
                                       │
                                       ↓
                               Reward Head  →  Predicted Reward
```

### Components

| Module | Path | Role |
|--------|------|------|
| `Encoder` | `src/models/encoder.py` | Obs → latent vector |
| `Decoder` | `src/models/decoder.py` | Latent → reconstructed obs |
| `TransitionModel` | `src/models/transition.py` | (z, a) → z' |
| `RewardModel` | `src/models/reward.py` | z → r |
| `WorldModel` | `src/models/world_model.py` | Combines above |

## Setup

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Verify GPU/MPS

```bash
python -c "
import torch
import gymnasium
print('torch:', torch.__version__)
print('gymnasium:', gymnasium.__version__)
if torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
elif torch.cuda.is_available():
    print('Device: CUDA', torch.cuda.get_device_name(0))
else:
    print('Device: CPU')
"
```

## Project Structure

```
tiny-world-model/
├── src/
│   ├── env/         # Environment wrappers
│   ├── models/      # Encoder, decoder, transition, reward
│   ├── training/    # Training loops, losses
│   └── eval/        # Evaluation and visualization
├── pyproject.toml
└── README.md
```
