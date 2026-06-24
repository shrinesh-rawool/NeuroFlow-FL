from env.traffic_env import TrafficEnv
from agents.ppo_agent import train_ppo


def train():

    # Training environment
    env = TrafficEnv("network/simulation.sumocfg")

    # Separate evaluation environment — required by EvalCallback in train_dqn()
    # Must be a different instance so evaluation episodes do not interfere
    # with the training environment's state
    eval_env = TrafficEnv("network/simulation.sumocfg")

    model = train_ppo(env, eval_env, timesteps=200_000)

    env.close()
    eval_env.close()

    return model


def run_with_preemption(model):

    # Emergency preemption is now handled inside TrafficEnv.step() automatically.
    # There is no need to manage it separately here — the env checks for
    # emergency vehicles and overrides the agent's action on every step.
    env = TrafficEnv("network/simulation.sumocfg")

    state, _ = env.reset()

    done = False

    while not done:

        # Agent predicts action — env.step() will internally override it
        # if an emergency vehicle is detected (via EmergencyPreemption)
        action, _ = model.predict(state, deterministic=True)

        state, reward, done, truncated, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    print("Training PPO agent...")
    model = train()

    print("Running simulation with trained PPO agent...")
    run_with_preemption(model)
    
    
    
    