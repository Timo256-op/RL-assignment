# RL/eval/evaluate.py
import os
import sys
import numpy as np
import argparse
from stable_baselines3 import PPO

# --- Ensure the environment module is found ---
# Add the parent directory (RL_assigment) to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from env.teacher_alloc_env import TeacherAllocEnv

# --- Optional: Suppress the non-critical Gym warning for a cleaner console ---
import warnings
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning,
    message=".*Gym has been unmaintained since 2022.*"
)

# --- Configuration ---
# Adjusted path to go up one directory (from 'eval' to 'RL_assigment') and then into 'saved_models'
DEFAULT_MODEL_PATH = os.path.join("..", "saved_models", "ppo_teacher_alloc.zip")

def evaluate(model_path: str, n_episodes: int = 100):
    """Loads a model and evaluates its performance on the environment."""
    
    # Check if the model exists using the absolute path for clarity in the error message
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    
    if not os.path.exists(full_path):
        print(f"[ERROR] Model not found at path: {full_path}")
        return [], []
    
    print(f"Loading model from: {full_path}")
    try:
        # Load the model
        model = PPO.load(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return [], []
    
    # Create the environment instance
    env = TeacherAllocEnv()
    
    rewards = []
    coverages = []

    print(f"Evaluating over {n_episodes} episodes...")
    
    for i in range(n_episodes):
        # Handle env.reset() return value for compatibility
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result
        terminated = False
        truncated = False
        ep_reward = 0
        covs = []
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            
            # env.step() returns 5 values: obs, reward, terminated, truncated, info
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except AttributeError as e:
                if "'numpy.random._generator.Generator' object has no attribute 'randint'" in str(e):
                    print("[ERROR] Detected use of numpy's new Generator API in your environment. 'randint' is not available. Please update your environment code to use 'integers' instead of 'randint'.")
                    print("For example, replace: rng.randint(a, b) with rng.integers(a, b)")
                    return [], []
                else:
                    raise
            ep_reward += reward
            covs.append(info.get("coverage_rate", 0.0))
            
        rewards.append(ep_reward)
        coverages.append(np.mean(covs) if covs else 0.0)

    print("\n--- Evaluation Results ---")
    print(f"Episodes evaluated: {n_episodes}")
    print(f"Mean Episode Reward: {np.mean(rewards):.4f} (Std: {np.std(rewards):.4f})")
    print(f"Mean Coverage Rate: {np.mean(coverages):.4f} (Std: {np.std(coverages):.4f})")
    
    return rewards, coverages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model zip file. Default: {DEFAULT_MODEL_PATH}"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run for evaluation. Default: 100"
    )
    args = parser.parse_args()
    
    # Debug: Print parsed arguments
    print(f"[DEBUG] model_path: {args.model_path}")
    print(f"[DEBUG] episodes: {args.episodes}")
    # Run evaluation
    evaluate(args.model_path, args.episodes)