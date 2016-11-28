import gym
import tensorflow as tf
import numpy as np

def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()

def build_weights(input_size, hidden_size, output_size):
  init = tf.constant(np.random.randn(input_size,
    hidden_size)/np.sqrt(input_size), dtype=tf.float32)
  W0 = tf.get_variable('W0', initializer=init)
  b0 = tf.Variable(tf.constant(0., shape=(hidden_size,)))
  init2 = tf.constant(1e-4*np.random.randn(hidden_size, output_size),
      dtype=tf.float32)
  W1 = tf.get_variable('W1',initializer=init2)
  b1 = tf.Variable(tf.constant(0., shape=(output_size,)))

  return {'W0':W0, 'b0':b0, 'W1':W1, 'b1':b1}

class LOLAgent(object):
  def __init__(self, obs_space, action_space, **usercfg):
    nO = obs_space.shape[0]
    nA = action_space.n

    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.05,
        nhid_p = 20,
        nhid_v = 20,
        episode_max_length = 100,
      )

    self.config.update(usercfg)

    self.ob = tf.placeholder(tf.float32, (None, nO), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')

    # policy network
    with tf.variable_scope('policy'):
      net = build_weights(nO, self.config['nhid_p'], nA)
      self.pol_prob = tf.nn.softmax(tf.matmul(tf.tanh(tf.matmul(
            self.ob, net['W0']) + net['b0']), net['W1']) + net['b1'])

      self.pol_params = list(net.values())

    # value network
    with tf.variable_scope('value'):
      net = build_weights(nO, self.config['nhid_v'], 1)
      self.val = tf.matmul(tf.tanh(tf.matmul(
            self.ob, net['W0']) + net['b0']), net['W1']) + net['b1']
      self.val_params = list(net.values())

    ac_oh = tf.reshape(tf.one_hot(self.ac, nA), (-1, nA))
    masked_prob_na = tf.reduce_sum(ac_oh * self.pol_prob, reduction_indices=1)
    score = tf.mul(tf.log(masked_prob_na), self.rew - self.val)
    self.value_loss = tf.nn.l2_loss(self.rew-self.val)

    self.pol_grads = tf.gradients(score, self.pol_params)
    self.val_grads = tf.gradients(self.value_loss, self.val_params)

    opt = tf.train.RMSPropOptimizer(self.config['lr'], momentum=0.9,
        epsilon=1e-9)
    self.pol_train = opt.minimize(-score)
    self.val_train = opt.minimize(self.value_loss)

    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def act(self, ob):
    prob = self.sess.run(self.pol_prob, {self.ob:np.reshape(ob, (1, -1))})
    action = categorical_sample(prob)
    return action

  def learn(self, env):
    ob = env.reset()
    for iteration in range(self.config['n_iter']):
      obs = []
      acts = []
      rews = []
      for t in range(self.config['t_max']):
        a = self.act(ob)
        ob, rew, done, _ = env.step(a)
        obs.append(ob)
        acts.append(a)
        rews.append(rew)
        env.render()
        if done:
          ob = env.reset()
          break

      if done:
        R = 0
      else:
        R = self.sess.run(self.val, {self.ob:np.reshape(obs[-1], (1,4))})

      RR = np.zeros((t+1, 1))
      RR[-1, 0] = R
      for i in reversed(range(t)):
        RR[i, 0] = R
        R = rews[i] + self.config['gamma'] * R

      obs = np.reshape(obs, (-1,4))
      acts = np.reshape(acts, (-1,1))
      _, _, val, val_l = self.sess.run([self.pol_train, self.val_train, self.val, self.value_loss],
          {self.ob:obs, self.ac:acts, self.rew:RR})

def main():
    env = gym.make("CartPole-v0")
    agent = LOLAgent(env.observation_space, env.action_space,
        episode_max_length=3000, lr=0.002, hidden_size=30, gamma=0.999,
        n_iter=10000)
    agent.learn(env)

if __name__ == "__main__":
    main()
