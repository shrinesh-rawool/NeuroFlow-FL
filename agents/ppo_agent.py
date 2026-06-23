# agents/ppo_agent.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


def create_ppo(env):

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="results/logs/PPO"
    )

    return model


def train_ppo(env, eval_env, timesteps=200000):

    model = create_ppo(env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="results/models/ppo_best/",
        log_path="results/logs/eval_ppo/",
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)

    os.makedirs("results/models", exist_ok=True)
    model.save("results/models/ppo_traffic_final")

    return model


def load_ppo(model_path, env):
    return PPO.load(model_path, env=env)