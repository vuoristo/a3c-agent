import gym
import tensorflow as tf
import numpy as np
import threading
from collections import deque

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
  b0 = tf.get_variable('b0', initializer=tf.constant(0., shape=(hidden_size,)))
  init2 = tf.constant(1e-4*np.random.randn(hidden_size, output_size),
      dtype=tf.float32)
  W1 = tf.get_variable('W1',initializer=init2)
  b1 = tf.get_variable('b1', initializer=tf.constant(0., shape=(output_size,)))

  return {'W0':W0, 'b0':b0, 'W1':W1, 'b1':b1}

class ThreadModel(object):
  def __init__(self, nO, nA, global_policy, global_value, config):
    self.ob = tf.placeholder(tf.float32, (None, nO), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')

    # policy network
    with tf.variable_scope('policy'):
      net = build_weights(nO, config['nhid_p'], nA)
      self.pol_prob = tf.nn.softmax(tf.matmul(tf.tanh(tf.matmul(
            self.ob, net['W0']) + net['b0']), net['W1']) + net['b1'])
      pol_names, pol_vars = zip(*net.items())

    # value network
    with tf.variable_scope('value'):
      net = build_weights(nO, config['nhid_v'], 1)
      self.val = tf.matmul(tf.tanh(tf.matmul(
            self.ob, net['W0']) + net['b0']), net['W1']) + net['b1']
      val_names, val_vars = zip(*net.items())

    ac_oh = tf.reshape(tf.one_hot(self.ac, nA), (-1, nA))
    masked_prob_na = tf.reduce_sum(ac_oh * self.pol_prob, reduction_indices=1)
    score = tf.mul(tf.log(tf.clip_by_value(masked_prob_na, 1.e-10, 1.0)), self.rew - self.val)
    value_loss = tf.nn.l2_loss(self.rew - self.val)

    # TODO: do we want to get the gradients from the optimizer or are there
    # better places to do that?
    opt = tf.train.RMSPropOptimizer(config['lr'], momentum=0.9,
        epsilon=1e-9)

    pg = opt.compute_gradients(score, pol_vars)
    self.pol_grads = {k: v for k, v in zip(pol_names, pg)}
    vg = opt.compute_gradients(value_loss, val_vars)
    self.val_grads = {k: v for k, v in zip(val_names, vg)}

    self.pol_updates = self.get_updates(global_policy, self.pol_grads,
        config['update_rate'])
    self.val_updates = self.get_updates(global_value, self.val_grads,
        config['update_rate'])

  def get_updates(self, global_net, grads, lr):
    updates = []
    for key, grad_entry in grads.items():
      W = global_net.get(key)
      grad = tf.clip_by_norm(grad_entry[0], 1.)
      l_to_g = W.assign(W - lr * grad)
      g_to_l = grad_entry[1].assign(W)
      updates += [l_to_g, g_to_l]

    return updates

class LOLAgent(object):
  def __init__(self, obs_space, action_space, **usercfg):
    self.nO = obs_space.shape[0]
    self.nA = action_space.n

    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.05,
        update_rate = 0.002,
        nhid_p = 20,
        nhid_v = 20,
        episode_max_length = 100,
        num_threads = 1,
      )

    self.config.update(usercfg)

    with tf.variable_scope('global_policy'):
      global_policy = build_weights(self.nO, self.config['nhid_p'], self.nA)
    with tf.variable_scope('global_value'):
      global_value = build_weights(self.nO, self.config['nhid_v'], 1)

    self.thr_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.nO, self.nA, global_policy,
            global_value, self.config)
        self.thr_models.append(thr_model)

    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def act(self, ob, model):
    prob = self.sess.run(model.pol_prob, {model.ob:np.reshape(ob, (1, -1))})
    action = categorical_sample(prob)
    return action

  def learning_thread(self, thread_id):
    env = gym.make("CartPole-v0")
    model = self.thr_models[thread_id]
    ob = env.reset()
    env_steps = 0
    ravg = deque(maxlen=100)
    for iteration in range(self.config['n_iter']):
      obs = []
      acts = []
      rews = []
      for t in range(self.config['t_max']):
        a = self.act(ob, model)
        ob, rew, done, _ = env.step(a)
        obs.append(ob)
        acts.append(a)
        rews.append(rew)
        env_steps += 1
        if done:
          ravg.append(env_steps)
          ob = env.reset()
          print('Thread: {} Steps: {} av100: {}'.format(
            thread_id, env_steps, np.mean(ravg)))
          env_steps = 0
          break

      if done:
        R = 0
      else:
        R = self.sess.run(model.val, {model.ob:np.reshape(obs[-1],
          (1,self.nO))})

      RR = np.zeros((t+1, 1))
      RR[-1, 0] = R
      for i in reversed(range(t)):
        RR[i, 0] = R
        R = rews[i] + self.config['gamma'] * R

      obs = np.reshape(obs, (-1,self.nO))
      acts = np.reshape(acts, (-1,1))
      _, _ = self.sess.run([model.pol_updates, model.val_updates],
          {model.ob:obs, model.ac:acts, model.rew:RR})

  def learn(self):
    ths = [threading.Thread(target=self.learning_thread, args=(i,)) for i in
        range(self.config['num_threads'])]

    for t in ths:
      t.start()

def main():
    env = gym.make("CartPole-v0")
    agent = LOLAgent(env.observation_space, env.action_space,
        episode_max_length=10000, update_rate=0.001, gamma=0.999, n_iter=10000)
    agent.learn()

if __name__ == "__main__":
    main()
