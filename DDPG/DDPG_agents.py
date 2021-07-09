import torch
import numpy as np

import torch.nn.functional as F

class DDPGAgent:
    def __init__(self, env, buffer, actor, critic, target_actor, target_critic, actor_optim, 
    critic_optim, noise_gen, device):
        
        self.env = env
        self.buffer = buffer
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.noise_gen = noise_gen
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

        # For logging purposes safe the actor and critic loss at each 
        # optimizer step
        self.actor_loss = 0.0
        self.critic_loss = 0.0

        # CPU oder GPU ("cuda")
        assert device in ["cpu", "cuda"], "Unknown device"
        self.device = device
    
    @torch.no_grad() # The action selection process is not needed to have a computational graph
    def select_noisy_action(self, state, min_action, max_action):

        """Action selection via the actor network. Noise gets added to the calculated
        actions to better explore"""

        state = torch.tensor(state).to(self.device)

        # Get action via actor
        action = self.actor(state.float()) # Action is in [-1,1] due to tanh in the net

        # Scale action to maximum value
        action *= max_action

        # Add noise to the action
        action += self.noise_gen.sample()

        # Clip the actions to the interval of the allowed action space
        action = torch.clip(action,min_action,max_action)

        return action

    def memorize(self, state, action, reward, n_state, done):
        self.buffer.add(state,action,reward, n_state, done)

    def play_step(self, action):
        n_state, reward, done, _ = self.env.step(action)
        return n_state, reward, done

    def calculate_loss_and_train(self, batch, gamma):
        state, action, reward, n_state, done = batch

        # Calculate current estimated Q-value
        q_val = self.critic(state, action)

        # Critic loss
        with torch.no_grad():
            # Calculate next action with target actor
            n_action = self.target_actor(n_state)

            # Calculate target Q-Value with the target critic
            target_Q = self.target_critic(n_state, n_action)

            # Targets if the last state was a terminal one then Q = 0
            target_Q[done] = 0.0

            target_Q = reward + gamma * target_Q

        critic_loss = F.mse_loss(q_val, target_Q)
        
        # Log the loss
        self.critic_loss = critic_loss.item()

        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss

        # Freeze critic
        for p in self.critic.parameters():
            p.requires_grad = False

        # Get actions from from states of the replay buffer using the most recent actor
        current_actions = self.actor(state)

        # Calc actor loss
        # Negative sign for gradient ascent.
        # Becomes aprox performance measure by chain rule
        actor_loss = -self.critic(state,current_actions).mean()

        self.actor_loss = actor_loss.item()

        actor_loss.backward()
        self.actor_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True


    @torch.no_grad()
    def update_target_networks(self, tau):
        # Actor update
        for target_params, main_params in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_params.data.copy_(tau * main_params.data + (1-tau) * target_params.data)

        # Critic update    
        for target_params, main_params in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_params.data.copy_(tau * main_params.data + (1-tau) * target_params.data)