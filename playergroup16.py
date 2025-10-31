from dicewars import player
from PPOagent import PPOAgent

import os
import numpy as np

class Player(player.Player):

    def __init__(self): # Initialize the player

        self.playername = 'Group 16'

        # Path of training weigths
        self.actor_path='actor.weights.h5'
        self.critic_path='critic.weights.h5'

        self.PPOAgent = PPOAgent() # Initialize the PPO agent

        # Check if the player is already trained
        if os.path.exists(self.actor_path) and os.path.exists(self.critic_path):
            self.load_model_weights()
        else:   # Train the player
            print("No pre-trained weights found. Starting training...")
            self.PPOAgent.run()

    def load_model_weights(self):
        """
        Loads the weights of the actor and critic networks
        """
        self.PPOAgent.actor.load_weights(self.actor_path)
        self.PPOAgent.critic.load_weights(self.critic_path)

        print(f"Weights loaded: Actor -> {self.actor_path}, Critic -> {self.critic_path}")

    def get_attack_areas(self, grid, match_state):
        """
        Selects an attack action during the game
        """
        from_player = match_state.player
        player_areas = match_state.player_areas
        area_num_dice = match_state.area_num_dice

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

        # If there is no possible action, return None as an action
        if len(possible_attacks) == 0:
            possible_attacks = [None]
            return None

        # Construct the state 
        state_features = self.PPOAgent.state_descriptor(grid, match_state)

        # Add batch dimension to the state (state_dim,) -> (1, state_dim)
        state = np.expand_dims(state_features, axis=0)
        probs = np.array(self.PPOAgent.actor(state).numpy()[0])  # Predict probabilities as a NumPy array

        # Get valid probabilities for only valid actions
        valid_probs = np.zeros(len(possible_attacks))
        for i, action in enumerate(possible_attacks):
            valid_probs[i] = probs[self.PPOAgent.action_map[action]]  

        # Normalize probabilities
        if valid_probs.sum() == 0:
            valid_probs += 1e-6  # Prevent zero probabilities
        else:
            valid_probs /= valid_probs.sum()

        # Select Greedy action (max probability)
        selected_index = np.argmax(valid_probs)  
        selected_action = possible_attacks[selected_index]

        # Debugging
        """
        print("Valid probabilities:", valid_probs)
        print(f"Possible Actions: {possible_attacks}")
        print(f"Selected Action: {selected_action}, Probability: {valid_probs[selected_index]}")
        """
        return selected_action
    
    
    





