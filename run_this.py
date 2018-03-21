# CartPole-v0
import gym
import math
from brain import DQNAgent

#def modify_reward(observation, x_th, theta_th):
def modify_reward(observation):
    x, v = observation
    reward = math.sin(3 * x)*.45+.55
    return reward

if __name__ == "__main__":
    # New environment
    env = gym.make('MountainCar-v0')
    #env = env.unwrapped # To access inner variables like x_threshold and theta_threshold_radians

    # Record the video
    directory = 'videos/'
    env = gym.wrappers.Monitor(env, directory, force=True)

    # New agent
    agent = DQNAgent(
        n_actions = env.action_space.n,
        space_shape = env.observation_space.shape[0],
        batch_size = 128,
        learning_rate = 0.005,
        epsilon = 0.9,
        gamma = 0.9,
        target_replace_iter = 256,
        replay_memory_size = 8000,
        output_graph = True,
        restore_tf_variables = False,
    )

    for i_episode in range(400):
        # Initial observation
        observation = env.reset()

        t = 0
        score = 0
        while True:
            env.render()

            # Choose action based on observation
            action = agent.choose_action(observation)

            # Take action
            observation_, reward, done, info = env.step(action)

            # Modify the reward
            reward = modify_reward(observation_)

            # Update score
            score += reward

            # Store transition
            agent.store_transition(observation, action, reward, observation_, done)

            # Learn from experience
            agent.learn()

            # Swap observations
            observation = observation_
            t += 1

            if done:
                print 'Ep: {0:3d} steps: {1:3d} score: {2:4.2f}'.format(i_episode, t, score)
                break

    agent.save('models/model.ckpt')
