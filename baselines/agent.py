import tensorflow as tf
import numpy as np

from baselines.actor import actor
from baselines.critic import critic

class agent(object):
    def __init__(self):
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=3e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=3e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_param = 0.2
        self.gamma = 0.95
        self.landa = 0.8
        self.epsilon = 0.1
        self.last_entropy = 100

        print(tf.__version__)

    def act(self, state):
        if np.random.uniform(size=1, low=0.0, high=1.0) < self.epsilon:
            return np.random.uniform(size=4, low=-1.0, high=1.0)
        else:
            prob = self.actor(np.array([state]))
            mu = prob[0].numpy()
            std = prob[1].numpy()
            dist = tf.compat.v1.distributions.Normal(mu, std + 1e-10, True)
            action = dist.sample()
            return action.numpy()[0]

    def save(self):
        vars = self.actor.trainable_variables
        savee = []
        for var in vars:
            savee.append(var.numpy().tolist())

        f = open("actor_cp.save", "w")
        f.write(str(savee))
        f.close()

        vars = self.critic.trainable_variables
        savee = []
        for var in vars:
            savee.append(var.numpy().tolist())

        f = open("critic_cp.save", "w")
        f.write(str(savee))
        f.close()

    # can be optimized
    def preprocess(self, mini_batch):
        #observation, value, action, reward, done
        states = []
        values = []
        returns = []
        actions = []
        advs = []
        probs = []

        g = 0
        size = len(mini_batch)
        idx = size - 1
        for el in mini_batch[::-1]:
            next_val = mini_batch[idx + 1][1] if idx < size - 1 else 0
            delta = el[3] + self.gamma * next_val - el[1]
            advs.append(delta)
            g = el[3] + self.gamma * g

            states.append(el[0])
            values.append(el[1])
            returns.append(g)
            actions.append(el[2])
            probs.append(el[5])
            idx -= 1

        probs.reverse()
        advs.reverse()
        states.reverse()
        values.reverse()
        returns.reverse()
        actions.reverse()

        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-10)

        return states, values, returns, actions, advs, probs

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        p = tf.compat.v1.distributions.Normal(loc=probs[0], scale=probs[1] + 1e-10, validate_args=True)
        probability = p.prob(actions)        

        log_prob = tf.math.log(probability + 1e-10)
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,log_prob)))
        self.last_entropy = entropy

        sur1 = []
        sur2 = []
        
        for pb, t, op, act in zip(probability, adv, old_probs, actions):
            o_p = tf.compat.v1.distributions.Normal(loc=op[0], scale=op[1] + 1e-10, validate_args=True)
            old_probability = o_p.prob(act)
            
            t =  tf.constant(t)
            op =  tf.constant(old_probability) + 1e-5
            ratio = tf.math.divide(pb, op)
            s1 = tf.math.multiply(ratio, t)
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param), t)
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        if np.isnan(entropy) or np.isinf(entropy):
            entropy = 0
        
        min = tf.math.minimum(sr1, sr2)
        mean = tf.reduce_mean(min)
        loss = tf.math.negative(mean) - 0.1 * entropy

        if np.isnan(loss.numpy()).any() or np.isinf(loss.numpy()).any():
            print("nan")

        return loss

    def learn(self, mini_batch):
        [states, values, returns, actions, advs, old_probs] = self.preprocess(mini_batch)
        
        states = tf.constant(states)
        returns = tf.constant(returns, dtype=tf.float32)
        returns = tf.reshape(returns, (returns.shape[0], 1))

        with tf.GradientTape(persistent=True) as t, tf.GradientTape(persistent=True) as t2:
            p = self.actor(states)
            v = self.critic(states)
            td = tf.math.subtract(returns, v)
            c_loss = tf.norm(td)
            a_loss = self.actor_loss(p, actions, advs, old_probs, c_loss)

        grads1 = t.gradient(a_loss, self.actor.trainable_variables)
        grads2 = t2.gradient(c_loss, self.critic.trainable_variables)

        for g in grads1:
            if np.isnan(g.numpy()).any():
                print("nan value")
                return 100, 100

        for g in grads2:
            if np.isnan(g.numpy()).any():
                print("nan")
                return 100, 100

        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        
        del t
        del t2
        return c_loss, a_loss
        