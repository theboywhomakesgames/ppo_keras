from baselines.actor import actor
from collections import deque
import tensorflow as tf
import numpy as np
from datetime import datetime
from baselines.agent import agent

class worker(object):
    def __init__(self, env):
        self.agent = agent()
        self.time_step = 0
        self.replay_buffer = deque(maxlen=1200)
        self.priorities = deque(maxlen=1200)
        self.learn_after = 1200
        self.reduce_after = 2000
        self.last_best = -1e10
        self.learn_each = 10
        self.epochs = 32
        self.super_learn = 1200
        self.episode_length = 200
        self.env = env

    def set_up_logging(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "logs/func/%s" % stamp
        self.writer = tf.summary.create_file_writer(logdir)

    def work(self):
        observation = self.env.reset()

        # self.agent.actor.set_inputs(observation)
        # self.agent.critic.set_inputs(observation)
        
        self.set_up_logging()
        for i_episode in range(1, 2000000):
            observation = self.env.reset()

            # loading weights and biases
            # action = self.agent.actor(np.array([observation]))
            # value = self.agent.critic(np.array([observation]))
            # self.agent.load_wabs()

            mini_batch = []
            acc_reward = 0
            td = 0
            val_sum = 0
            g_sum = 0
            for t in range(self.episode_length):
                action = self.agent.act(observation)
                p = self.agent.actor(np.array([observation]))
                new_observation, reward, done = self.env.step(action)

                acc_reward += reward

                value = self.agent.critic(np.array([observation]))
                new_value = self.agent.critic(np.array([new_observation]))

                g = (reward + self.agent.gamma * new_value)
                val_sum += value
                g_sum += g

                mini_batch.append([observation, value, action, reward, done, p])
                observation = new_observation
                self.time_step += 1

                if done:
                    break
            
            td = (g_sum - val_sum).numpy()[0][0]
            g_sum = g_sum.numpy()[0][0]
            val_sum = val_sum.numpy()[0][0]

            print(str(i_episode) + " reward: " + str(acc_reward) + " td: " + str(td) + " g: " + str(g_sum)+ " v: " + str(val_sum))
            self.replay_buffer.append(mini_batch)

            if i_episode % self.reduce_after == 0:
                self.agent.reduce_learning_rate()
                # self.reduce_after -= 10
                # if self.reduce_after < 5:
                #     self.reduce_after = 5
                # print("epsilon: " + str(self.agent.epsilon))

            if i_episode >= self.learn_after and i_episode % self.learn_each == 0:
                self.agent.epsilon *= 0
                if(self.agent.epsilon < 1e-5):
                    self.agent.epsilon = 0

                c_loss = 0
                a_loss = 0

                if i_episode % self.learn_after == 0:
                    print("super learn")
                    for i in range(self.super_learn):
                        print("learning epoch: " + str(i))
                        idx = np.random.randint(low=0, high=len(self.replay_buffer), size=1)
                        choice = self.replay_buffer[idx[0]]
                        [closs, aloss] = self.agent.learn(choice)

                        a_loss += aloss
                        c_loss += closs

                    a_loss /= self.super_learn
                    c_loss /= self.super_learn

                else:
                    print("learn")
                    for i in range(self.epochs):
                        print("learning epoch: " + str(i))
                        idx = np.random.randint(low=0, high=len(self.replay_buffer), size=1)
                        choice = self.replay_buffer[idx[0]]
                        [closs, aloss] = self.agent.learn(choice)

                        a_loss += aloss
                        c_loss += closs

                    a_loss /= self.epochs
                    c_loss /= self.epochs
                
                with self.writer.as_default():
                    tf.summary.scalar("eposide reward", acc_reward, step=self.time_step)
                    tf.summary.scalar("policy loss", a_loss, step=self.time_step)
                    tf.summary.scalar("value loss", c_loss, step=self.time_step)
                    tf.summary.scalar("leaning rate", self.agent.learning_rate, step=self.time_step)
                    tf.summary.scalar("td", td, step=self.time_step)
                self.writer.flush()
                
                if i_episode % self.reduce_after == 0:
                    self.epochs -= 1
                    if(self.epochs < 10):
                        self.epochs = 10

                # if i_episode % 100 == 0:
                #     self.episode_length += (500 / self.episode_length)
                #     self.episode_length = int(self.episode_length)
                #     if self.episode_length > 2000:
                #         self.episode_length = 2000

                # self.learn_after -= (self.learn_after ** 2) / 600
                # self.learn_after = int(self.learn_after)

                # if(self.learn_after < 10):
                #     self.learn_after = 10

                if acc_reward > self.last_best:
                    self.last_best = acc_reward
                    self.agent.save()

        self.env.close()
