# CartPole RL with Cross-Entropy Method

A minimal reinforcement learning project that trains an agent to balance a 
cart-pole system — built with only NumPy (no gymnasium/MuJoCo dependency).

## What it does

- **`CartPoleEnv`** — simulates cart-pole physics from scratch using the 
  standard control equations (cart position/velocity, pole angle/angular velocity).
- **`LinearPolicy`** — a simple linear policy that maps the 4D state to a 
  left/right push action.
- **`train_cem`** — trains the policy using the **Cross-Entropy Method (CEM)**: 
  sample a population of random policies, keep the top performers ("elites"), 
  and narrow the search distribution around them each generation.

## Results

Converges to a near-perfect score (~499/500 reward) within ~8 generations.

## Run it

```bash
python3 cartpole_cem.py
