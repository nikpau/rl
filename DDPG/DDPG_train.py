from torch import optim
from DDPG_agents import DDPGAgent
from DDPG_networks import Actor, Critic
from DDPG_noise import GaussianNoise
from DDPG_replay_buffers import ReplayBufferUniform
import gym 
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np 
import torch.nn as nn
import copy


# Hyperparameter
ENV_NAME = "Pendulum-v0"
MAX_ACTION = 2
MIN_ACTION = -2

GAMMA = 0.99
TAU = 0.001
REWARD_BOUND = 90
LEARNING_RATE_ACTOR = 10e-4
LEARNING_RATE_CRITIC = 10e-3
BUFFER_SIZE = 100000
BATCH_SIZE = 64
MAX_EPISODE_LENGTH = 199


if __name__=="__main__":

    env = gym.make(ENV_NAME)
    state = env.reset()

    # Init replay buffer
    buffer = ReplayBufferUniform(env.action_space.shape[0],env.observation_space.shape[0],
    BUFFER_SIZE, BATCH_SIZE, "cpu")
    # Actor and critic networks
    actor = Actor(env.observation_space.shape[0],env.action_space.shape[0])
    actor_target = copy.deepcopy(actor)

    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])
    critic_target = copy.deepcopy(critic)

    # Optimizers
    actor_optim = optim.Adam(actor.parameters(),lr=LEARNING_RATE_ACTOR)
    critic_optim = optim.Adam(critic.parameters(),lr=LEARNING_RATE_CRITIC)

    # Gaussian Noise
    g_noise = GaussianNoise(env.action_space.shape[0])

    # Agent
    agent = DDPGAgent(env, buffer, actor, critic, 
    actor_target, critic_target, actor_optim, critic_optim, g_noise, "cpu")

    # SummaryWriter
    writer = SummaryWriter()

    episode_reward_list = []
    episode_reward = 0.0
    episode_step_counter = 0
    episode_counter = 0
    iter_no = 0

    while True:

        episode_step_counter += 1
        iter_no += 1
        
        # Select action 
        action = agent.select_noisy_action(state, MIN_ACTION,MAX_ACTION)

        # Make a step in the environment
        new_state, reward, done, _ = env.step(action)

        # Reward per episode
        episode_reward += reward

        # Set done to True if max steps per episode is reached
        done = True if episode_step_counter%MAX_EPISODE_LENGTH==0 else done

        # Write the transition tuple into the replay buffer
        agent.memorize(state, action, reward, new_state, done)

        # Net new state to current state
        state = new_state

        # End of episode handling
        if done:
            state = env.reset()
            episode_reward_list.append(episode_reward)
            mean_reward = np.mean(episode_reward_list[-100:])
            writer.add_scalar("Episode Reward", episode_reward)
            writer.add_scalar("Iteration", iter_no)
            writer.add_scalar("Mean Reward", mean_reward)
            print(f"Iteration {iter_no} | Episode {episode_counter} | Episode Reward {episode_reward:.2f} | Mean Reward {mean_reward:.2f}")
            print("Fitting...", end="\r")
            episode_counter += 1
            episode_step_counter = 0
            episode_reward = 0

        if buffer.len() < BUFFER_SIZE:
            print("Filling buffer...", end="\r")
            continue

        if mean_reward > REWARD_BOUND:
            print(f"Solved in {episode_counter} episodes.")

        critic_optim.zero_grad()
        actor_optim.zero_grad()
        batch = buffer.sample()
        agent.calculate_loss_and_train(batch, GAMMA)
        agent.update_target_networks(TAU)