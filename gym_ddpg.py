import filter_env
from ddpg import *
import gc
gc.enable()
import time

ENV_NAME = 'gym_finger:Finger-v0'
EPISODES = 100000
TEST = 10

def main():
    env = gym.make(ENV_NAME)#filter_env.makeFilteredEnv(gym.make(ENV_NAME))
#    env.setRealTimeSimulation()
    agent = DDPG(env)
    timestep_limit = 300
    # env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        print("********RESET*********")
        print("episode:",episode)
        # Train
        for step in xrange(timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            print(reward, action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            env.render()
            if done:
                break

        # Testing:
        if episode % 20 == 0 and episode > 0:
            print("# # # # TEST # # # #")
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(timestep_limit):
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    env.render()
                    if done:
                        break
            ave_reward = total_reward/TEST
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
            print("# # # # # # # # # # #")

        # Saving the model
        if episode % 20 == 0 and episode > 0:
            agent.save(episode)

    # env.monitor.close()

if __name__ == '__main__':
    main()
