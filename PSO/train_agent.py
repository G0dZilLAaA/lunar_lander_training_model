import gymnasium as gym  # Import Gymnasium for reinforcement learning environment
import numpy as np  # Import NumPy for numerical operations
import argparse  # Import argparse to handle command-line arguments
import os  # Import os for file operations
from multiprocessing import Pool  # Import Pool for parallel processing
from functools import partial  # Import partial to create functions with preset arguments
from sklearn.cluster import KMeans  # Import KMeans for clustering initial swarm positions

# Policy function: Maps observation to action using a linear policy
def policy_action(params, observation):
    """Compute action from observation using a linear policy."""
    W = params[:32].reshape(8, 4)  # Extract weight matrix (8x4)
    b = params[32:].reshape(4)  # Extract bias vector (4,)
    logits = np.dot(observation, W) + b  # Compute logits (linear transformation)
    return np.argmax(logits)  # Select action with highest score

# Evaluate a policy's fitness by running multiple episodes
def evaluate_policy(params, episodes=10, render=False):
    """Evaluate policy using multiple episodes and return fitness score."""
    env = gym.make("LunarLander-v3", render_mode="human" if render else "rgb_array")  # Create environment
    total_reward, landings, crashes = 0.0, 0, 0  # Initialize tracking variables
    
    for _ in range(episodes):  # Run multiple episodes
        obs, _ = env.reset()  # Reset environment, get initial observation
        done = False  # Episode termination flag
        episode_reward = 0.0  # Track cumulative reward for this episode
        
        while not done:  # Run until episode ends
            action = policy_action(params, obs)  # Get action from policy
            obs, reward, terminated, truncated, _ = env.step(action)  # Apply action in environment
            episode_reward += reward  # Accumulate reward
            done = terminated or truncated  # Episode ends if terminated or truncated
        
        total_reward += episode_reward  # Accumulate total reward
        landings += 1 if episode_reward > 200 else 0  # Count successful landings
        crashes += 1 if episode_reward < -100 else 0  # Count crashes
    
    env.close()  # Close environment

    # Compute average reward, landing rate, and crash rate
    avg_reward = total_reward / episodes
    landing_rate = landings / episodes
    crash_rate = crashes / episodes

    # Compute fitness function (balance reward, landings, and crashes)
    fitness = avg_reward + 100 * landing_rate - 50 * crash_rate
    return fitness, avg_reward  # Return fitness score and average reward

# Evaluate a batch of positions using parallel processing
def evaluate_batch(positions, episodes=10):
    """Evaluate a batch of candidate policies in parallel."""
    with Pool() as pool:  # Create multiprocessing pool
        eval_func = partial(evaluate_policy, episodes=episodes)  # Wrap evaluation function
        results = pool.map(eval_func, positions)  # Evaluate all positions in parallel

    fitness_scores, rewards = zip(*results)  # Extract fitness scores and rewards
    return np.array(fitness_scores), np.array(rewards)  # Return as arrays

# Load policy helper function
def load_existing_policy(filename):
    if os.path.exists(filename):
        print(f"Loaded existing policy from {filename} for initialization")
        return np.load(filename)
    return None

# Particle Swarm Optimization (PSO) to optimize policy parameters
def particle_swarm_optimization(swarm_size=50, max_iterations=350, lower_bound=-5, upper_bound=5, policy_file=None):
    """Optimize Lunar Lander policy parameters using Particle Swarm Optimization."""
    param_size = 36  # 32 for weight matrix + 4 for bias vector

    # Check for existing saved policy
    saved_params = load_existing_policy(policy_file) if policy_file else None
    
    if saved_params is not None and saved_params.shape == (param_size,):
        print("Initializing swarm with saved policy")
        # Initialize swarm around saved policy with some perturbation
        positions = np.tile(saved_params, (swarm_size, 1))
        noise = np.random.normal(0, 0.5, (swarm_size, param_size))
        positions += noise
        positions = np.clip(positions, lower_bound, upper_bound)
    else:
        # Fallback to original KMeans initialization
        print("No valid saved policy found, using random initialization")
        init_points = np.random.uniform(lower_bound, upper_bound, (swarm_size * 2, param_size))
        kmeans = KMeans(n_clusters=swarm_size, n_init=5).fit(init_points)  
        positions = kmeans.cluster_centers_  # Set initial particle positions

    velocities = np.random.uniform(-0.5, 0.5, (swarm_size, param_size))  # Initialize velocities

    pbest_positions = positions.copy()  # Initialize personal best positions
    pbest_scores = np.full(swarm_size, -np.inf)  # Set initial best scores to negative infinity
    gbest_position = None  # Initialize global best position
    gbest_score = -np.inf  # Initialize global best score
    stagnation_counter = 0  # Track if optimization is stagnating

    # PSO hyperparameters (inertia weight and acceleration coefficients)
    w_max, w_min = 0.9, 0.4  # Inertia weight range
    c1_max, c1_min = 2.5, 0.5  # Cognitive acceleration coefficient range
    c2_max, c2_min = 0.5, 2.5  # Social acceleration coefficient range

    # Display progress header
    print(f"{'Iteration':^10} | {'Global Best':^12} | {'Avg Reward':^12} | {'Std Dev':^12}")
    print("-" * 50)

    for iteration in range(max_iterations):
        # Evaluate the current swarm
        fitness, rewards = evaluate_batch(positions)

        # Update personal bests
        better_mask = fitness > pbest_scores
        pbest_positions[better_mask] = positions[better_mask]
        pbest_scores[better_mask] = fitness[better_mask]

        # Update global best
        max_fitness = fitness.max()
        if max_fitness > gbest_score:
            gbest_score = max_fitness
            gbest_position = positions[fitness.argmax()].copy()
            stagnation_counter = 0  # Reset stagnation counter
        else:
            stagnation_counter += 1  # Increment stagnation counter

        # Print progress every 5 iterations
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        if iteration % 5 == 0:
            print(f"{iteration+1:^10} | {gbest_score:^12.2f} | {mean_reward:^12.2f} | {std_reward:^12.2f}")

        # Adjust PSO hyperparameters over time
        progress = iteration / max_iterations
        w = w_max - (w_max - w_min) * progress
        c1 = c1_max - (c1_max - c1_min) * progress
        c2 = c2_min + (c2_max - c2_min) * progress

        # Update velocities using PSO equations
        r1, r2 = np.random.rand(2, swarm_size, param_size)
        velocities = (w * velocities +
                      c1 * r1 * (pbest_positions - positions) +
                      c2 * r2 * (gbest_position - positions))
        velocities = np.clip(velocities, -2.0, 2.0)  # Limit velocity magnitude

        # Introduce small noise if stagnation occurs
        if stagnation_counter > 15:
            noise = np.random.normal(0, 0.1, positions.shape)
            positions = np.clip(positions + noise, lower_bound, upper_bound)
            stagnation_counter = 0  # Reset stagnation counter

        # Update positions and ensure they remain within bounds
        positions += velocities
        positions = np.clip(positions, lower_bound, upper_bound)

    return gbest_position  # Return best parameters found

# Train a policy using PSO and save it to a file
def train_and_save(filename, swarm_size=50, max_iterations=350, lower_bound=-5, upper_bound=5):
    """Train a policy using PSO and save the best parameters."""
    best_params = particle_swarm_optimization(swarm_size, max_iterations, lower_bound, upper_bound, policy_file=filename)
    np.save(filename, best_params)  # Save to file
    print(f"\nBest policy saved to {filename}")
    return best_params

# Load a saved policy from file
def load_policy(filename):
    """Load trained policy parameters from a file."""
    if os.path.exists(filename):
        print(f"Loaded best policy from {filename}")
        return np.load(filename)
    print(f"File {filename} does not exist.")
    return None

# Play the game using a trained policy
def play_policy(params, episodes=10):
    """Play Lunar Lander using a trained policy and render results."""
    fitness, avg_reward = evaluate_policy(params, episodes=episodes, render=True)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f} (Fitness: {fitness:.2f})")

# Command-line interface for training and playing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Lunar Lander policy using enhanced PSO.")
    parser.add_argument("--train", action="store_true", help="Train a new policy.")
    parser.add_argument("--play", action="store_true", help="Play a saved policy.")
    parser.add_argument("--filename", type=str, default="best_policy_21.npy", help="Policy file.")
    args = parser.parse_args()

    if args.train:
        train_and_save(args.filename)
    elif args.play:
        params = load_policy(args.filename)
        if params is not None:
            play_policy(params)
    else:
        print("Use --train to train or --play to test.")
