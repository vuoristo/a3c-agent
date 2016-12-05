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

def build_network(input_size, hidden_1, hidden_2, output_size):
  init = tf.uniform_unit_scaling_initializer()
  W0 = tf.get_variable('W0', shape=(input_size, hidden_1), initializer=init)
  b0 = tf.get_variable('b0', initializer=tf.constant(0., shape=(hidden_1,)))

  W_softmax = tf.get_variable('W_softmax', shape=(hidden_1, output_size),
    initializer=init)
  b_softmax = tf.get_variable('b_softmax', initializer=tf.constant(0.,
    shape=(output_size,)))

  W_linear = tf.get_variable('W_linear', shape=(hidden_1, 1), initializer=init)
  b_linear = tf.get_variable('b_linear', initializer=tf.constant(0.,
    shape=(1,)))

  pol_vars = [W0, b0, W_softmax, b_softmax]
  val_vars = [W_linear, b_linear, W0, b0]

  return pol_vars, val_vars

class ThreadModel(object):
  def __init__(self, nO, nA, global_policy, global_value, pol_grad_msq,
               val_grad_msq, config):
    self.ob = tf.placeholder(tf.float32, (None, nO), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')
    self.lr = tf.placeholder(tf.float32, name='lr')

    pol_vars, val_vars = build_network(nO, config['hidden_1'],
      config['hidden_2'], nA)

    h_1 = tf.tanh(tf.nn.bias_add(tf.matmul(
      self.ob, pol_vars[0]), pol_vars[1]))

    self.pol_prob = tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_1, pol_vars[2]),
      pol_vars[3]))
    self.val = tf.nn.bias_add(tf.matmul(h_1, val_vars[0]), val_vars[1])

    ac_oh = tf.reshape(tf.one_hot(self.ac, nA), (-1, nA))
    masked_prob_na = tf.reduce_sum(ac_oh * self.pol_prob, reduction_indices=1)
    score = tf.mul(tf.log(tf.clip_by_value(
      masked_prob_na, 1.e-10, 1.0)), self.rew - self.val)
    value_loss = tf.nn.l2_loss(self.rew - self.val)

    self.pol_grads = tf.gradients(score, pol_vars)
    self.val_grads = tf.gradients(value_loss, val_vars)

    self.pol_updates = self.get_updates(global_policy, pol_vars,
        self.pol_grads, pol_grad_msq)
    self.val_updates = self.get_updates(global_value, val_vars,
        self.val_grads, val_grad_msq)

    self.dbg_var = self.pol_updates[0]

  def get_updates(self, global_vars, local_vars, grads, grad_msq, momentum=0.9,
                  epsilon=1e-9):
    updates = []
    for Wg, Wl, grad, msq in zip(global_vars, local_vars, grads, grad_msq):
      grad = tf.clip_by_norm(grad, 10.)

      # compute rmsprop update per variable
      ms_update = momentum * msq + (1. - momentum) * tf.pow(grad, 2)
      gradient_update = -self.lr * grad / tf.sqrt(ms_update + epsilon)

      # apply updates to global variables, copy back to local variables
      l_to_g = Wg.assign_add(gradient_update)
      g_to_l = Wl.assign(Wg)

      updates += [l_to_g, g_to_l, ms_update]

    return updates

class LOLAgent(object):
  def __init__(self, obs_space, action_space, **usercfg):
    self.nO = obs_space.shape[0]
    self.nA = action_space.n

    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.01,
        min_lr = 0.002,
        lr_decay_steps = 10000,
        hidden_1 = 20,
        hidden_2 = 5,
        episode_max_length = 100,
        num_threads = 1,
      )

    self.config.update(usercfg)

    with tf.variable_scope('global_network') as global_scope:
      pol_vars, val_vars = build_network(self.nO, self.config['hidden_1'],
        self.config['hidden_2'], self.nA)
      pol_grad_msq = [tf.Variable(np.zeros(var.get_shape(), dtype=np.float32))
          for var in pol_vars]
      val_grad_msq = [tf.Variable(np.zeros(var.get_shape(), dtype=np.float32))
          for var in val_vars]

    self.thr_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.nO, self.nA, pol_vars,
          val_vars, pol_grad_msq, val_grad_msq, self.config)
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
    lr = self.config['lr']
    lr_decay_step = (lr - self.config['min_lr'])/self.config['lr_decay_steps']
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
        if lr > self.config['min_lr']:
          lr -= lr_decay_step
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
      episode_max_length=10000, gamma=0.99, n_iter=1000000,
      num_threads=4, t_max=5, min_lr=0.0001, lr_decay_steps=30000)
    agent.learn()

if __name__ == "__main__":
    main()
