"""
CartPole Balancing with the Cross-Entropy Method (CEM)
--------------------------------------------------------
A minimal, dependency-free (only numpy) reinforcement learning project.

WHAT'S HAPPENING:
1. `CartPoleEnv` simulates a cart-and-pole system using real physics
   equations (the same dynamics used in classic control / MuJoCo-style
   tasks): a pole balanced on a cart, which we push left or right.
2. `LinearPolicy` maps the 4D state (cart position, cart velocity,
   pole angle, pole angular velocity) to an action (push left/right)
   using a simple weighted sum.
3. The Cross-Entropy Method (CEM) is a simple, effective RL algorithm:
   - Sample a batch of random policies (weight vectors) from a
     Gaussian distribution.
   - Run each one in the environment, measure how long it balances
     the pole (the "reward").
   - Keep the best-performing policies ("elites").
   - Update the distribution's mean/std toward those elites.
   - Repeat. Over generations, the distribution converges on
     weights that balance the pole well.

Run it with: python3 cartpole_cem.py
"""

import numpy as np


class CartPoleEnv:
    """Physics simulation of a cart-pole system (no external deps)."""

    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5  # half the pole's length
        self.force_mag = 10.0
        self.dt = 0.02  # seconds per simulation step

        # Episode ends if these limits are exceeded
        self.angle_limit = 12 * (np.pi / 180)  # ~12 degrees
        self.position_limit = 2.4

        self.max_steps = 500
        self.state = None
        self.steps = 0

    def reset(self):
        # Start near upright, with small random perturbation
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        """action: 0 (push left) or 1 (push right)"""
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta, sintheta = np.cos(theta), np.sin(theta)

        temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * costheta / self.total_mass

        # Euler integration
        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = bool(
            x < -self.position_limit
            or x > self.position_limit
            or theta < -self.angle_limit
            or theta > self.angle_limit
            or self.steps >= self.max_steps
        )
        reward = 1.0 if not done else 0.0  # +1 for every step still balanced
        return self.state.copy(), reward, done


class LinearPolicy:
    """Maps a 4D state to a binary action using a linear combination."""

    def __init__(self, weights):
        self.weights = weights  # shape (4,)

    def act(self, state):
        return 1 if np.dot(self.weights, state) > 0 else 0


def evaluate(weights, env, episodes=1):
    """Run a policy in the environment and return average total reward."""
    policy = LinearPolicy(weights)
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
    return total_reward / episodes


def train_cem(
    n_generations=25,
    population_size=40,
    elite_frac=0.2,
    n_dims=4,
    init_std=1.0,
):
    env = CartPoleEnv()
    mean = np.zeros(n_dims)
    std = np.ones(n_dims) * init_std
    n_elite = max(1, int(population_size * elite_frac))

    best_weights = mean.copy()
    best_reward = -np.inf

    for gen in range(n_generations):
        # Sample a population of candidate weight vectors
        population = np.random.randn(population_size, n_dims) * std + mean
        rewards = np.array([evaluate(w, env, episodes=3) for w in population])

        # Select elites (top performers) and refit the distribution
        elite_idx = rewards.argsort()[-n_elite:]
        elites = population[elite_idx]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3  # small epsilon avoids collapse to 0

        gen_best_reward = rewards.max()
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_weights = population[rewards.argmax()]

        print(f"Generation {gen + 1:2d}/{n_generations} | "
              f"avg reward: {rewards.mean():6.1f} | best so far: {best_reward:6.1f}")

    return best_weights, best_reward


if __name__ == "__main__":
    np.random.seed(0)
    print("Training a linear policy to balance CartPole using CEM...\n")
    weights, reward = train_cem()

    print(f"\nBest policy found. Weights: {weights}")
    print("Evaluating best policy over 10 episodes...")

    env = CartPoleEnv()
    final_score = evaluate(weights, env, episodes=10)
    print(f"Average reward over 10 test episodes: {final_score:.1f} / 500")
