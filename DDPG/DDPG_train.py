from torch import optim
from DDPG_agents import DDPGAgent
from DDPG_networks import Actor, Critic
from DDPG_noise import GaussianNoise
from DDPG_replay_buffers import ReplayBufferUniform
import gym 
from torch.utils.tensorboard import SummaryWriter
import csv
import numpy as np 
import torch.nn as nn
import copy

"""TODO:
    - Implement a logger and multithreading"""
# torch.set_num_threads(15)

# Hyperparameter
ENV_NAME = "MountainCarContinuous-v0"
MAX_ACTION = 5
MIN_ACTION = -5

NOISE_SIGMA = 0.3

GAMMA = 0.99
TAU = 0.001
REWARD_BOUND = 90
LEARNING_RATE_ACTOR = 10e-4
LEARNING_RATE_CRITIC = 10e-3
BUFFER_SIZE = 100000
BATCH_SIZE = 64
MAX_EPISODE_LENGTH = 200 


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
    g_noise = GaussianNoise(env.action_space.shape[0], 0, NOISE_SIGMA)

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
    iter_list = []
    actor_loss = []
    critic_loss = []
    mean_reward_list = []

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
        done = True if episode_step_counter % MAX_EPISODE_LENGTH==0 else done

        # Write the transition tuple into the replay buffer
        agent.memorize(state, action, reward, new_state, done)

        # Net new state to current state
        state = new_state

        # End of episode handling
        if done:
            state = env.reset() # Reset env
            episode_reward_list.append(episode_reward)


            episode_counter += 1
            episode_step_counter = 0
            episode_reward = 0

        # Do not begin training until buffer is filled 
        # (May be abandoned as training can already begin with filling buffer)
        if buffer.len() < BUFFER_SIZE:
            print(f"Filling buffer {iter_no} / {BUFFER_SIZE}" , end="\r")
            continue

        # Logging
        if iter_no % MAX_EPISODE_LENGTH == 0:
            mean_reward = np.mean(episode_reward_list[-100:]) # Calculate a mean from the last 100 rewards
            mean_reward_list.append(mean_reward)
            iter_list.append(iter_no)
            actor_loss.append(agent.actor_loss)
            critic_loss.append(agent.critic_loss)

            print(f"Iteration {iter_no} | Episode {episode_counter} | Episode Reward {episode_reward:.2f} | Mean Reward {mean_reward:.2f}")

            if iter_no % 10000 == 0:
                rows = zip(iter_list, mean_reward_list, actor_loss, critic_loss)
                with open("DDPG/" + ENV_NAME + "_log.csv", "w") as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerow(["iter_no","mean_reward","actor_loss", "ctitc_loss"])
                    for row in rows:
                        wr.writerow(row)

        # TODO: Implement other mechanism as the boundary of 0 will never be met. 
        # Maybe safe the net every n iterations / episodes
        if mean_reward > REWARD_BOUND:
            print(f"Solved in {episode_counter} episodes.")

        # Stdout prints
        print("Fitting...", end="\r")

        critic_optim.zero_grad()
        actor_optim.zero_grad()
        batch = buffer.sample() # Sample BATCH_SIZE transitions
        agent.calculate_loss_and_train(batch, GAMMA)
        agent.update_target_networks(TAU) # Polyak update