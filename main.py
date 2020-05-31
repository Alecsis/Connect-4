import gym
import gym_connect4


if __name__ == "__main__":
    # Build the environment
    env = gym.make('gym_connect4:connect4-v0')

    # Create Agents

    # Close environemtn
    env.close()