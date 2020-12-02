import gym
from collections import deque
import tensorflow as tf
import numpy as np
from datetime import datetime
from baselines.agent import agent

class worker(object):
    def __init__(self):
        self.agent = agent()
        self.env = gym.make('BipedalWalker-v3')
        self.time_step = 0
        self.replay_buffer = deque(maxlen=50000)
        self.learn_after = 10
        self.reduce_after = 50
        self.save_each = 2000
        self.epochs = 10
        self.episode_length = 1000

    def set_up_logging(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "logs/func/%s" % stamp
        self.writer = tf.summary.create_file_writer(logdir)

    def work(self):
        self.set_up_logging()
        for i_episode in range(1, 2000000):
            observation = self.env.reset()
            mini_batch = []
            acc_reward = 0
            for t in range(self.episode_length):
                self.env.render()
                action = self.agent.act(observation)
                p = self.agent.actor(np.array([observation]))
                new_observation, reward, done, info = self.env.step(action)

                acc_reward += reward
                value = self.agent.critic(np.array([observation]))
                mini_batch.append([observation, value, action, reward, done, p])
                observation = new_observation
                self.time_step += 1

                if done:
                    break

            print(str(i_episode) + " reward: " + str(acc_reward))
            self.replay_buffer.append(mini_batch)

            if i_episode % self.reduce_after == 0:
                self.agent.epsilon *= 0.5
                if(self.agent.epsilon < 1e-5):
                    self.agent.epsilon = 0

                self.reduce_after -= 10
                if self.reduce_after < 5:
                    self.reduce_after = 5
                print("epsilon: " + str(self.agent.epsilon))

            if i_episode > self.learn_after:
                c_loss = 0
                a_loss = 0

                for i in range(self.epochs):
                    print("learning epoch: " + str(i))
                    idx = np.random.randint(low=0, high=len(self.replay_buffer), size=1)
                    choice = self.replay_buffer[idx[0]]
                    [closs, aloss] = self.agent.learn(choice)

                    a_loss += aloss
                    c_loss += closs

                a_loss /= 64
                c_loss /= 64
                
                with self.writer.as_default():
                    tf.summary.scalar("eposide reward", acc_reward, step=self.time_step)
                    tf.summary.scalar("policy loss", a_loss, step=self.time_step)
                    tf.summary.scalar("value loss", c_loss, step=self.time_step)
                    tf.summary.scalar("policy_entr", self.agent.last_entropy, step=self.time_step)
                    tf.summary.scalar("epsilon", self.agent.epsilon, step=self.time_step)
                self.writer.flush()

                self.epochs -= 1
                if(self.epochs < 1):
                    self.epochs = 1

                # self.episode_length += (500 / self.episode_length)
                # self.episode_length = int(self.episode_length)
                # if self.episode_length > 2000:
                #     self.episode_length = 2000

                self.learn_after -= (self.learn_after ** 2) / 600
                self.learn_after = int(self.learn_after)

                if(self.learn_after < 10):
                    self.learn_after = 10

            if i_episode % self.save_each == 0:
                self.agent.save()

        self.env.close()
