```markdown
# Lunar Lander AI Training using Genetic Algorithm and Particle Swarm Optimization

This project implements AI training for the **Lunar Lander-v3** environment using **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**. The trained agent learns to land the spacecraft safely while maximizing rewards.

## 🚀 Overview

Reinforcement Learning (RL) techniques like **Neural Networks with Backpropagation** require extensive computation and gradient-based optimization. Instead, we use **Genetic Algorithms** and **Particle Swarm Optimization**, which are nature-inspired optimization techniques that work without explicit gradient calculations.

The training involves evolving a population of policies to find the best parameters for controlling the lander.

## 🔬 Training Algorithms

### 1️⃣ Genetic Algorithm (GA)

GA is an evolutionary approach that mimics natural selection:
- **Selection**: The best-performing policies (high rewards) are chosen.
- **Crossover**: New policies are created by combining parents' traits using **Simulated Binary Crossover (SBX)**.
- **Mutation**: Small random changes are introduced using **Polynomial Mutation**.
- **Elitism**: The best policies are carried forward to the next generation.

### 2️⃣ Particle Swarm Optimization (PSO)

PSO optimizes policies based on swarm intelligence:
- Each particle represents a candidate solution (policy parameters).
- **Velocity Update**: Each particle moves in search space using personal and global best solutions.
- **Position Update**: The new position is influenced by inertia, cognitive, and social components.
- **Exploration and Exploitation**: Balances searching new areas and refining good solutions.

## 📂 Project Structure

```
📂 LunarLander-Optimization
 ├── train_agent.py  # Main script for training and evaluation
 ├── best_policy.npy  # Saved trained policy
 ├── README.md  # Project documentation
 ├── requirements.txt  # Dependencies
```

## 📌 Installation & Setup

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/LunarLander-Optimization.git
   cd LunarLander-Optimization
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Train the agent:
   - Using Genetic Algorithm:
     ```sh
     python train_agent.py --train --filename best_policy.npy
     ```
   - Using Particle Swarm Optimization:
     ```sh
     python train_agent.py --train --filename best_policy_pso.npy
     ```

4. Play with the trained policy:
   ```sh
   python train_agent.py --play --filename best_policy.npy
   ```

## 📊 Results

The trained agent achieves an average reward of **above 200** after training, indicating a successful landing in most cases.

| Algorithm | Avg Reward | Convergence Speed |
|-----------|------------|------------------|
| GA        | ~220      | Medium           |
| PSO       | ~210      | Fast             |

## 🎯 Future Work

- Implementing **Deep Reinforcement Learning** (DQN, PPO).
- Hyperparameter tuning for GA and PSO.
- Multi-objective optimization for stability and efficiency.

## 👨‍💻 Author
**Mohit Kumawat**  
