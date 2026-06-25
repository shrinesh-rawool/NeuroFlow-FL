import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback


def create_dqn(env, log_dir="results/logs/default"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,           # size of weight update step
        buffer_size=50000,              # size of replay buffer (how many past experiences to store)
        learning_starts=5000,           # was 1000 — wait for buffer to fill ~10% before training
        batch_size=64,                  # number of random samples per training step
        gamma=0.99,                     # how much to value future rewarch vs. intermediate reward (0.99 = 1% discount per step)
        train_freq=4,                   # train the network every 4 environment step
        target_update_interval=1000,    # number of steps to sync the network
        exploration_fraction=0.3,       # decay epsilon over first 30% of total timesteps
        exploration_final_eps=0.05,     # settle at 5% random action after decay
        verbose=1,                      
        device=device,
        tensorboard_log="results/logs/"
    )

    return model


def train_dqn(env, eval_env, timesteps=200000):

    model = create_dqn(env)

    # EvalCallback tests the agent on a separate env every eval_freq steps,
    # saves the best-performing model automatically, and logs eval reward to TensorBoard
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="results/models/best/",
        log_path="results/logs/eval/",
        eval_freq=10_000,       # evaluate every 10,000 environment steps
        n_eval_episodes=3,      # average over 3 test episodes per evaluation
        deterministic=True,     # no exploration during evaluation
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)

    os.makedirs("results/models", exist_ok=True)

    model.save("results/models/dqn_traffic_final")

    return model


def load_dqn(model_path, env):

    model = DQN.load(model_path, env=env)

    return model