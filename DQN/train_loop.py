from agents import NoisyAgent
from replay_buffers import ReplayBufferUniform
from networks import NoisyNetwork
from torch.utils.tensorboard import SummaryWriter
import gym
import torch
import numpy as np
import torch.optim as optim


# HYPERPARAMETERS ---------------------------------

# Env
ENV_NAME = "CartPole-v0"
ACTIONS = {0: 0, 1: 1}
BUFFER_SIZE = 10000
REWARD_BOUND = 195

# Training
GAMMA = 0.99
BATCH_SIZE = 128
TARGET_NET_UPDATE = 150
LEARNING_RATE = 0.001
N_STEPS = 1


# Train until done.
if __name__ == "__main__":
    
    # Initialize env
    env = gym.make(ENV_NAME)
    state = env.reset() #first state
    
    # Init Replay Buffer    
    buffer = ReplayBufferUniform(buffer_size=BUFFER_SIZE)

    # Init networks
    net = NoisyNetwork(env.observation_space.shape,env.action_space.n)
    tgt_net = NoisyNetwork(env.observation_space.shape,env.action_space.n)
    
    # Optimizer init
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # Init agent
    agent = NoisyAgent(env,net,tgt_net, buffer, double=False)
    
    # Init Logger
    writer = SummaryWriter() # This is what uses the torch.utils.tensorboard btw 

    rewards_list = []
    episode_reward = 0.0
    iter_no = 0
    episode_no = 0

    # Actual training loop
    while True:
        
        iter_no += 1
        
        # Play n steps and return a transition tuple
        state, action, reward, new_state, done = agent.play_n_steps(state,N_STEPS,GAMMA)

        # Add transition to replay buffer
        agent.memorize(state,action,reward, new_state,done)
        
        state = new_state
        episode_reward += reward
        
        if done:
            episode_no += 1
            rewards_list.append(episode_reward)
            mean_reward = np.mean(rewards_list[-100:])
            print(f"Iteration: {iter_no}. Episode: {episode_no}. Mean reward: {mean_reward:.2f}.")

            # TODO: implement Summary Writer
            writer.add_scalar("Reward per iteration", episode_reward, iter_no)
            writer.add_scalar("Reward per episode", episode_reward, episode_no)
            writer.add_scalar("Mean Reward at iteration", mean_reward, iter_no)
            
            
            episode_reward = 0.0
            state = env.reset()
            
        if buffer.len() < BUFFER_SIZE:
            continue
        
        if iter_no % TARGET_NET_UPDATE == 0:
            agent.update_target_net()
        
        if mean_reward > REWARD_BOUND:
            print("SOLVED")
            torch.save(net.state_dict(), "DQN/" + ENV_NAME + "_noisy_weights.dat")
            break
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = agent.calculate_loss(batch,N_STEPS,GAMMA)
        loss_t.backward()
        optimizer.step()
        
        