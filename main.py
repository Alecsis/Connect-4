import gym
import gym_connect4
from agents.random_agent import RandomAgent


if __name__ == "__main__":
    # Build environment
    print("[.] Build Environment")
    env = gym.make('gym_connect4:connect4-v0')
    
    # Create random agent
    print("[.] Create Random Agent")
    agent = RandomAgent(env.action_space, env.observation_space)

    print(env.action_space.n)

    # Init environment
    done = False
    obs = env.reset()
    
    # Run game
    print("[.] Running game")
    while not done:
        obs, reward, done, info = env.step(agent.get_action(obs))

    # Final render
    print("[+] Done.")
    print("Infos: ", info)
    print("Final board: ")
    env.render()

    # Close environment
    env.close()