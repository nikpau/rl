from agents import EpsGreedyAgent
from replay_buffers import ProportionalPERBuffer_noHeap
from networks import NormalNetwork
from torch.utils.tensorboard import SummaryWriter
import gym
import torch
import numpy as np
import torch.optim as optim


# HYPERPARAMETERS ---------------------------------

# Env
ENV_NAME = "MountainCar-v0"
ACTIONS = {0: 0, 1: 1, 2: 2}
BUFFER_SIZE = 5000
REWARD_BOUND = -110

# Training
GAMMA = 0.999
BATCH_SIZE = 128
TARGET_NET_UPDATE = 256
LEARNING_RATE = 0.01
N_STEPS = 5

# For e-greedy strategy
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_FINAL = 0.001

# PER Buffer 
ALPHA = 0.5
BETA_INIT = 0.5
BETA_INC = 0.0001

# Train until done.
if __name__ == "__main__":
    
    # Initialize env
    env = gym.make(ENV_NAME)
    state = env.reset() #first state
    
    # Init Replay Buffer    
    buffer = ProportionalPERBuffer_noHeap(buffer_size=BUFFER_SIZE, alpha=ALPHA,beta_init=BETA_INIT,
                                          beta_inc=BETA_INC,state_dim=env.observation_space.shape[0],
                                          action_dim=env.action_space.n)

    # Init networks
    net = NormalNetwork(env.observation_space.shape,env.action_space.n)
    tgt_net = NormalNetwork(env.observation_space.shape,env.action_space.n)
    
    # Optimizer init
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # Init agent
    agent = EpsGreedyAgent(env,net,tgt_net, buffer, double=True)
    
    # Init Logger
    writer = SummaryWriter() # This is what uses the torch.utils.tensorboard btw 

    rewards_list = []
    episode_reward = 0.0
    iter_no = 0
    episode_no = 0
    
    # Epsilon for trainig
    eps = EPSILON_START
    
    # Actual training loop
    while True:
        
        iter_no += 1
        eps = max(EPSILON_FINAL, eps * EPSILON_DECAY)
        
        
        # Play n steps and return a transition tuple
        state, action, reward, new_state, done = agent.play_n_steps(state,N_STEPS,eps,GAMMA)

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
        
        