import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

def create_dqn(env, run_name="dqn_default"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        device=device,
        tensorboard_log=f"results/logs/{run_name}/",
    )
    return model

def train_dqn(env, eval_env, timesteps=200000, run_name="dqn_default"):
    model = create_dqn(env, run_name=run_name)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"results/models/{run_name}/best/",
        log_path=f"results/logs/{run_name}/eval/",
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)

    os.makedirs(f"results/models/{run_name}", exist_ok=True)
    model.save(f"results/models/{run_name}/final_model")

    return model

def load_dqn(model_path, env):
    return DQN.load(model_path, env=env)