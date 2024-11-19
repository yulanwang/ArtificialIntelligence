import time
import numpy as np
from vis_gym import *

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial 
	   configuration and taking actions until a terminal state is reached.
	2. Keep track of gameplay history in an appropriate format for each of the episodes.
	3. From gameplay history, estimate the probability of victory against each of the guards when taking the fight action.

	Some important notes:

		a. Keep in mind that given some observation [(X,Y), health, guard_in_cell], a fight action is only meaningful if the 
		   last entry corresponding to guard_in_cell is nonzero.

		b. Upon taking the fight action, if the player defeats the guard, the player is moved to a random neighboring cell with 
		   UNCHANGED health. (2 = Full, 1 = Injured, 0 = Critical).

		c. If the player loses the fight, the player is still moved to a random neighboring cell, but the health decreases by 1.

		d. Your player might encounter the same guard in different cells in different episodes.

		e. All interaction with the environment must be done using the env.step() method, which returns the next
		   observation, reward, done (Bool indicating whether terminal state reached) and info. This method should be called as 
		   obs, reward, done, info = env.step(action), where action is an integer representing the action to be taken.

		f. The env.reset() method resets the environment to the initial configuration and returns the initial observation. 
		   Do not forget to also update obs with the initial configuration returned by env.reset().

		g. To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		   For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		   will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		h. To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		   Example usage below. This function should be called after every action.

		   if gui_flag:
		       refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the np array, P which contains four float values, each representing the probability of defeating guards 1-4 respectively.

'''

def estimate_victory_probability(num_episodes=100000):
	"""
    Probability estimator

    Parameters:
    - num_episodes (int): Number of episodes to run.

    Returns:
    - P (numpy array): Empirically estimated probability of defeating guards 1-4.
    """
	P = np.zeros(len(env.guards))

	'''

	YOUR CODE HERE
	'''

    # Initialize counters for wins and encounters against each guard
	win_count = np.zeros(len(env.guards))
	encounter_count = np.zeros(len(env.guards))
	state_history = {}  # Hash values to record the gameplay history of each state

    # Loop over the number of episodes
	for episode in range(num_episodes):
        # Reset the environment at the start of each episode
		obs, _, done, _ = env.reset()
		flag_done = 0
		while not done:
			state_hash = hash(obs)
			if state_hash not in state_history:
				state_history[state_hash] = {'encounters': 0, 'victories': 0}
            # Check if the player encounters a guard
			guard = obs['guard_in_cell']
			if guard:
                # Convert 'G1', 'G2', etc. to index 0, 1, 2, 3
				guard_id = int(guard[-1]) - 1

                # Update encounter count for the current guard
				state_history[state_hash]['encounters'] += 1
				encounter_count[guard_id] += 1

                # Simulate fight action
				fight_action = np.random.choice([4, 5])
                # fight_action = 4
				obs, reward, done, info = env.step(fight_action)

                # If reward is positive, the player won the fight
				if reward > 0:
					state_history[state_hash]['victories'] += 1
					win_count[guard_id] += 1
                    # If the player reaches the Critical health state, the variable flag_done set True.
                # If the player loses another fight, the game ends in defeat.
				if not flag_done and done and env.current_state['player_health'] == 'Critical':
					flag_done = 1
					done = False
					env.current_state['player_health'] = 'Injured'
			else:
                # Choose a movement action here. (UP, DOWN, LEFT, RIGHT)
				action = np.random.randint(0, 4)
                # Take the step in the environment
				obs, reward, done, info = env.step(action)
            # Refresh GUI if enabled
			if gui_flag:
				refresh(obs, reward, done, info)

    # Calculate the probability of victory against each guard
	P = np.divide(win_count, encounter_count, out=np.zeros_like(win_count), where=encounter_count != 0)

	return P

probabilities = estimate_victory_probability(num_episodes=10000)
print(f"Probabilities of defeating guards 1-4 are: {probabilities}")


