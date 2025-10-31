import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import numpy as np
import random
import os  

from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import DefaultPlayer, AgressivePlayer, RandomPlayer, WeakerPlayerAttacker, PassivePlayer


class PPOAgent:

    def __init__(self):     # Agent Initiallization

        # TRAIN PARAMETERS
        self.lr_actor   = 3e-4       # Learning rate for the policy network
        self.lr_critic  = 1e-3       # Learning rate for the value network
        self.gamma      = 0.99       # Discount factor for future rewards --- gamma = 1 --> no discount in future awards
        self.epsilon    = 0.4        # Parameter that controls the change between the old and the new policy (0.1, 0,2)
        
        self.e_explore = 0.6         # Exploration parameter 
        self.e_min     = 0.1        # Minimum e_explore value
        self.e_decay   = 0.99       # Decreasing rate of e_explore

        # GAME PARAMETERS
        self.grid_areas  = 30        # Number of areas of the grid
        self.num_players = 4         # Number of players of the game
        self.num_possible_grid_actions = self.grid_areas * self.grid_areas     # Number of all possible actions that can be played on the grid
        # Dimension of the input state for training
        self.state_dim = 30 + 4 + 4 + 4 + 4     # num_dices_neighbours_per_area + num_total_dices_per_player + 
                                                # num_areas_per_player + max_cluster_per_player + num_stock_per_player    
                                                
        # Build Policy (Actor) & Value (Critic) Networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # Optimizers for weights update
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)

        # Mapping actions to unique indices
        self.action_map = {}            # Maps (from_area, to_area) -> unique index
        self.reverse_action_map = {}    # Maps unique index -> (from_area, to_area)
        self.all_possible_actions = np.array([(i, j) for i in range(self.grid_areas)    # All possible actions of the grid as tuple (from_area, to_area)
                                                    for j in range(self.grid_areas)])   

        for index, action in enumerate(self.all_possible_actions):
            self.action_map[tuple(action)] = index          
            self.reverse_action_map[index] = tuple(action)  

    # Policy and Value Neural Networks functions
    def build_actor(self):
        """
        Builds the policy (actor) network ---> 4 layers
        Returns the probabilities of all possible actions of the grid for a given state.
        """
        inputs = Input(shape=(self.state_dim,))     
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(inputs)  # for overfitting
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self.num_possible_grid_actions, activation='softmax')(x)  # Probabilities distribution

        return Model(inputs, outputs)

    def build_critic(self):
        """
        Builds the value (critic) network ---> 4 layers
        Returns the value function (V(s)) of a state --> expected future rewards of a state.
        """
        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(inputs)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(1, activation=None)(x)  # Value estimation

        return Model(inputs, outputs)
    
    # Implements the Advantage function of the agent
    def compute_advantage(self, rewards, states):
        """
        Computes advantage function A(s, a) using GAE.
        """
        states = np.array(states, dtype=np.float32)
        values = self.critic.predict(states).flatten()  # Values prediction from critic network (1D array)

        # Debugging
        #print(f"Rewards: {rewards}")
        #print(f"Values: {values}")

        # Advantage Function using Generalized Advantage Estimation (GAE)
        GAE_lamda = 0.95    # lamda parameter
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            # TD Error
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t] 
            # Advantage function    
            advantages[t] = last_advantage = delta + self.gamma * GAE_lamda * last_advantage  
            

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages *= 10  # Scale the advantages

        # Debugging
        #print(f"Advantages: {advantages}")

        return advantages

    def train(self, states, actions, rewards, old_probs):
        """
        Trains the PPO Agent
        """
        states    = np.array(states, dtype=np.float32)
        actions   = tf.one_hot(actions, self.num_possible_grid_actions)
        rewards   = np.array(rewards, dtype=np.float32)
        old_probs = np.array(old_probs, dtype=np.float32)

        advantages = self.compute_advantage(rewards, states)  # Compute advantages

        # Update Critic Network (Value Estimation)
        with tf.GradientTape() as tape:     # Calculating gradients
            value_loss = tf.reduce_mean((rewards - tf.squeeze(self.critic(states))) ** 2)  # MSE loss (accuracy of prediction)
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)   # Compute Gradients
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, 5) # Gradients clipping
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))   # Gradients Update

        # Update Actor Network (Policy) using Clipped PPO Loss
        with tf.GradientTape() as tape:
            probs = self.actor(states)    
            new_probs = tf.reduce_sum(actions * probs, axis=1)
            log_ratio = tf.math.log(new_probs + 1e-8) - tf.math.log(old_probs + 1e-8)
            ratio = tf.exp(log_ratio)   # Actual ratio
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) # Clip ratio between the bounds
            loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))  # PPO loss

        actor_grads = tape.gradient(loss, self.actor.trainable_variables)   # Compute actor gradients
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 5)   # Gradients clipping
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))  # Gradients Update

        # Debugging
        """
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        print(f"Actor Weights (after update): {actor_weights[:1]}") 
        print(f"Critic Weights (after update): {critic_weights[:1]}")

        actor_grad_norm = tf.linalg.global_norm(actor_grads)
        critic_grad_norm = tf.linalg.global_norm(critic_grads)
        print(f"Actor Gradient Norm: {actor_grad_norm}, Critic Gradient Norm: {critic_grad_norm}")
        """

        print(f"Actor Loss: {loss.numpy()}, Critic Loss: {value_loss.numpy()}")

        return loss.numpy(), value_loss.numpy()  # Return actor and critic losses


    # Training function
    def run(self):
        """
        Trains the PPO agent for a number of episodes
        """
        # TRAINING PARAMETERS
        episodes = 2000    # Number of training games
        train_freq = 20    # Number of playing games before training

        # File for saving training results 
        # In train/ folder
        results_file = "train/train_results.txt"  
        os.makedirs("train", exist_ok=True)

        RENDER = False  # Show playing game

        # Open the file for writing 
        with open(results_file, "a") as log:

            log.write("------TRAIN PROCESS-----\n")

            # Tracking training information
            wins = 0
            rewards_per_game = 0
            rewards_history = []
            wins_history = []
            actor_losses = []
            critic_losses = []

            # Initialize memory for training
            state_memory, action_memory, reward_memory, prob_memory = [], [], [], []

            # Initialize players
            #other_players = [RandomPlayer(), RandomPlayer(), RandomPlayer()]
            other_players = [AgressivePlayer(), WeakerPlayerAttacker(), PassivePlayer()]
            players = [self] + other_players

            for episode in range(episodes):

                # Start a new game
                game = Game(num_seats=len(players))
                match = Match(game)
                grid, state = match.game.grid, match.state

                while True:  # Play the game until is finished

                    currentplayer = players[state.player]

                    if state.player == 0:  # PPO Agent

                        # Get the important features of the state
                        state_features = self.state_descriptor(grid, state)
                        # Get the action from the PPO agent
                        action, prob = self.get_attack_areas(grid, state, state_features)
                        # Play the action
                        grid, next_state = match.step(action)

                        if action is not None:
                            # Get the reward for the action
                            reward = self.get_reward(state, next_state)
                            rewards_per_game += reward

                            # Store the training data
                            state_memory.append(state_features)
                            action_memory.append(self.action_map[action])
                            reward_memory.append(reward)
                            prob_memory.append(prob)
                            
                        state = next_state
                    else:   # Opponents
                        action = currentplayer.get_attack_areas(grid, state)
                        grid, state = match.step(action)

                    # Game simulation
                    if RENDER:
                        match.render()

                    # Quit if game is finished
                    if state.winner != -1:
                        if state.winner == 0:
                            wins += 1
                        rewards_history.append(rewards_per_game)  # Sum the rewards at the end of the game
                        rewards_per_game = 0  # Reset rewards for the next game
                        break

                # Train the agent after every train_freq games
                if (episode + 1) % train_freq == 0: 

                    # Compute discounted rewards
                    discounted_rewards = []
                    cumulative_reward = 0
                    for reward in reversed(reward_memory[-len(state_memory):]): 
                        cumulative_reward = reward + self.gamma * cumulative_reward
                        discounted_rewards.insert(0, cumulative_reward)

                    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

                    # Get a random batch from state memory
                    batch = random.randint(10, len(state_memory))

                    states = random.sample(state_memory, batch)
                    actions = random.sample(action_memory, batch)
                    rewards = random.sample(discounted_rewards.tolist(), batch)  # Convert to list
                    probs = random.sample(prob_memory, batch)

                    actor_loss, critic_loss = self.train(states, actions, rewards, probs)

                    # Store losses
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                # Update exploration parameter after each game
                if self.e_explore > self.e_min:
                    self.e_explore *= self.e_decay
                self.e_explore = max(self.e_explore, self.e_min)    # Ensure epsilon is always greater then the min

                print(f"Episode {episode + 1}/{episodes} , Total Reward: {rewards_history[-1]}\n")
                wins_history.append(wins / (episode + 1))  # Store win rate

            print(f"Player 16 won {wins} / {episodes}, Win Rate: {wins / episodes}")
            # Training infos for analysis
            log.write(f"lr_actor: {self.lr_actor}, lr_critic: {self.lr_critic}, gamma: {self.gamma}, epsilon: {self.epsilon}, TRAIN RATE: {wins / episodes}\n")

        self.training_plots(actor_losses, critic_losses, wins_history, rewards_history)  # Plot training data
        self.save_model_weights()  # Save weights after training

    def state_descriptor(self, grid, state):
        """
        Creates a normalized state vector from the current state,
        extracting the features needed for training.
        """
        neighbours_states = []
        neighbours_areas = []

        # Find the areas that are neighbours of the player's areas
        for area in state.player_areas[0]:
            neighbours_areas.extend(grid.areas[area].neighbors)

        # List of neighboring areas
        neighbours_areas = list(set(neighbours_areas))  # Keep unique indices

        # Mask for neighbours
        mask = np.zeros_like(state.area_num_dice, dtype=np.float32)
        mask[neighbours_areas] = 1

        area_num_dice = np.array(state.area_num_dice, dtype=np.float32)
        num_dice_neighbours = area_num_dice * mask  # Get neighbour's dices

        # Prepare states features
        player_dices = np.array(state.player_num_dice, dtype=np.float32)
        player_areas_flat = np.array([len(areas) for areas in state.player_areas], dtype=np.float32) if state.player_areas else np.zeros(4, dtype=np.float32)
        player_max_size = np.array(state.player_max_size, dtype=np.float32) if state.player_max_size else np.zeros(4, dtype=np.float32)
        player_num_stock = np.array(state.player_num_stock, dtype=np.float32) if state.player_num_stock else np.zeros(4, dtype=np.float32)

        # Normalize features to [0 , 1]
        num_dice_neighbours /= (np.max(num_dice_neighbours) + 1e-8)  
        player_dices /= (np.max(player_dices) + 1e-8)                
        player_areas_flat /= (np.max(player_areas_flat) + 1e-8)     
        player_max_size /= (np.max(player_max_size) + 1e-8)         
        player_num_stock /= (np.max(player_num_stock) + 1e-8)        

        # Create the state list
        neighbours_states.append(num_dice_neighbours)   # Number of neighbour's dices ---> Length = 30
        neighbours_states.append(player_dices)          # Number of dices per player ---> Length = 4
        neighbours_states.append(player_areas_flat)     # Number of areas per player ---> Length = 4
        neighbours_states.append(player_max_size)       # Size of largest cluster per player ---> Length = 4
        neighbours_states.append(player_num_stock)      # Number of stock dices per player ---> Length = 4

        # Concatenate all features into a single array
        state_features = np.concatenate([np.atleast_1d(feature) for feature in neighbours_states]).astype(np.float32)

        return state_features

    # Selects an attack    
    def get_attack_areas(self, grid, state, state_features):
        """
        Returns an attack action based on the policy network 
        and its probability
        """
        from_player = state.player
        player_areas = state.player_areas
        area_num_dice = state.area_num_dice

        # Initialize possible attacks with a fallback action
        possible_attacks = []

        # Loop over all areas in possession of the current player
        for from_area in player_areas[from_player]:
            # Check if the area has more than 1 die
            if area_num_dice[from_area] > 1:
                # Loop over all neighbors of the current area
                for to_area in grid.areas[from_area].neighbors:
                    # Check if the neighbor belongs to another player
                    if to_area not in player_areas[from_player]:
                        possible_attacks.append((from_area, to_area))

        # If there is no possible action, return None for action
        if len(possible_attacks) == 0:
            possible_attacks = [None]
            return None, 0  # action = None, prob = 0

        # Select an action based on the policy
        selected_action, probs = self.select_action(state_features, possible_attacks)

        return selected_action, probs
    
    # Selects a valid action
    def select_action(self, state, possible_actions):
        """
        Returns a valid action based on the probabilities predictions
        of the policy network
        """
        state = np.expand_dims(state, axis=0)   # Add batch dimension (state_dim,) -> (1, state_dim)
        probs = np.array(self.actor(state).numpy()[0])  # Predict probabilities as a NumPy array

        # Get valid probabilities of only valid actions
        valid_probs = np.zeros(len(possible_actions))

        for i, action in enumerate(possible_actions):
            valid_probs[i] = probs[self.action_map[action]]

        # Normalize probabilities
        if valid_probs.sum() == 0:
            valid_probs += 1e-6  # Prevent zero probabilities
        else:
            valid_probs /= valid_probs.sum()

        # Select action
        # Epsilon-Greedy strategy for exploration
        if np.random.uniform(0, 1) < self.e_explore:
            selected_index = np.random.choice(len(possible_actions))  # Random index action 
        else:
            selected_index = np.argmax(valid_probs)  # Greedy index action (max probability)

        selected_action = possible_actions[selected_index]

        # Debugging
        """
        print("Valid probabilities:", valid_probs)
        print(f"Possible Actions: {possible_actions}")
        print(f"Selected Action: {selected_action}, Probability: {valid_probs[selected_index]}")
        """
        return selected_action, valid_probs[selected_index]  # Return action & probability

    def get_reward(self, state, next_state):
        """
        Gives a reward based on the current and the next state
        """
        reward = 0.0

        # Rewards criteria
        area_diff = next_state.player_num_areas[0] - state.player_num_areas[0]  # number of areas 
        dice_diff = next_state.player_num_dice[0] - state.player_num_dice[0]    # number of dice 
        stock_diff = next_state.player_num_stock[0] - state.player_num_stock[0] # number of stock dice

        reward += 1.0 * area_diff
        reward += 0.05 * dice_diff
        reward += 0.05 * stock_diff

        if state.winner == 0:   # Big reward when agent wins
            reward += 10.0
    
        # Debugging
        #print("Step Reward:", reward)

        return reward


    # Helpers functions for debugging and data handling
    # All data are stored in train/ folder

    def save_model_weights(self, actor_path='train/actor.weights.h5', critic_path='train/critic.weights.h5'):
        """
        Saves the weights of the actor and critic networks in train/ folder
        actor weights --> train/actor.weights.h5
        critic wights --> train/critic.weights.h5
        """
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

        print(f"Weights saved: Actor -> {actor_path}, Critic -> {critic_path}")


    def training_plots(self, actor_losses, critic_losses, wins_history, rewards_history):
        """
        Plots the data from the learning process and saves them in train/ folder

            Policy Loss Plot --> train/actor_loss_{lr_actor}.png
            Value Loss Plot --> train/critic_loss_{lr_critic}.png
            Winning Rate during training Plot --> train/wins_history_{lr_actor}.png
            Rewards during training Plot --> train/rewards_history_(lr_actor}.png
        """
        # Ensure the 'train' directory exists
        os.makedirs("train", exist_ok=True)

        # Plot and save actor loss
        actor_loss_filename = f"train/actor_loss_lr_{self.lr_actor}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses, label="Actor Loss", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss")
        plt.title(f"Actor Loss with Learning Rate {self.lr_actor}")
        plt.legend()
        plt.savefig(actor_loss_filename)  # Save
        plt.close()

        # Plot and save the critic loss
        critic_loss_filename = f"train/critic_loss_lr_{self.lr_critic}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(critic_losses, label="Critic Loss", color="red")
        plt.xlabel("Episode")
        plt.ylabel("Critic Loss")
        plt.title(f"Critic Loss with Learning Rate {self.lr_critic}")
        plt.legend()
        plt.savefig(critic_loss_filename)  # Save
        plt.close()

        # Plot and save the winning rate
        wins_history_filename = f"train/wins_history_lr_{self.lr_actor}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(wins_history, label="Winning Rate", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.title(f"Win Rate during training with Learning Rate {self.lr_actor}")
        plt.legend()
        plt.savefig(wins_history_filename)
        plt.close()
       
        # Plot and save the rewards during training

        rewards_history_filename = f"train/rewards_history_lr_{self.lr_actor}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_history, label="Raw Rewards", color="green", alpha=0.4)
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.title(f"Rewards with Learning Rate {self.lr_actor}")
        plt.legend()
        plt.grid(True)
        plt.savefig(rewards_history_filename)
        plt.close()










