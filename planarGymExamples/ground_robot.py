import gym
import groundRobots
import numpy as np

obstacles = False


def main():
    n_arm = 10
    env = gym.make("ground-robot-arm-vel-v0", render=True, n_arm=n_arm, dt=0.01)
    #defaultAction = np.array([1.1, 0.50, -0.1, 0.1])
    #defaultAction = np.array([0.5, 0.5, 0.5, -0.5, 0, 0])
    defaultAction = np.ones(n_arm+2)*0.5
    # env = gym.make('ground-robot-vel-v0', render=True, dt=0.01)
    # defaultAction = np.array([1.0, 0.0])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        # ob = env.reset(
        #     pos=np.array([0.0, 1.0, 0.6 * np.pi, 0.0, 0.0]), vel=np.array([0.1, 0.0, 0.1, 0.1])
        # )
        ob = env.reset(
            pos=np.array([0.0, 0.0, 0.0, 0.5*np.pi, 0.0, 0, 0]), vel=np.array([0.0, 0.0, 0.0, 0.0, 0, 0])
        )
        if obstacles:
            from planarGymExamples.obstacles import sphereObst1, sphereObst2, dynamicSphereObst1

            env.addObstacle(sphereObst1)
            env.addObstacle(sphereObst2)
            env.addObstacle(dynamicSphereObst1)
        # ob = env.reset(pos=np.array([0.0, 1.0, 0.6 * np.pi]), vel=np.array([0.1, 0.0]))
        print("Starting episode")
        for i in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
