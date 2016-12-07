import gym
import tensorflow as tf
import numpy as np
import threading
from collections import deque

def categorical_sample(prob):
  """
  Sample from categorical distribution,
  specified by a vector of class probabilities
  """
  prob = np.asarray(prob)
  csprob = np.cumsum(prob)
  return (csprob > np.random.rand()).argmax()

class ThreadModel(object):
  def __init__(self, input_size, output_size, global_network, config):
    self.ob = tf.placeholder(tf.float32, (None, input_size), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')
    self.lr = tf.placeholder(tf.float32, name='lr')

    W0_size = config['hidden_1']

    # Model variables
    with tf.variable_scope('shared_variables') as sv_scope:
      init = tf.uniform_unit_scaling_initializer()
      W0 = tf.get_variable('W0', shape=(input_size, W0_size), initializer=init)
      b0 = tf.get_variable('b0', initializer=tf.constant(0., shape=(W0_size,)))

    with tf.variable_scope('policy') as pol_scope:
      W_softmax = tf.get_variable('W_softmax', shape=(W0_size, output_size),
        initializer=init)
      b_softmax = tf.get_variable('b_softmax',
        initializer=tf.constant(0., shape=(output_size,)))

    with tf.variable_scope('value') as val_scope:
      W_linear = tf.get_variable('W_linear', shape=(W0_size, 1),
        initializer=init)
      b_linear = tf.get_variable('b_linear',
        initializer=tf.constant(0., shape=(1,)))

    # Variable collections for update computations
    shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
      scope=sv_scope.name)
    self.pol_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
      scope=pol_scope.name) + shared_vars
    self.val_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
      scope=val_scope.name) + shared_vars

    # Global model stores the RMSProp moving averages.
    # Thread models contain the evaluation and update logic.
    if global_network is None:
      self.pol_grad_msq = [tf.Variable(np.zeros(var.get_shape(),
        dtype=np.float32)) for var in self.pol_vars]
      self.val_grad_msq = [tf.Variable(np.zeros(var.get_shape(),
        dtype=np.float32)) for var in self.val_vars]
    else:
      h_1 = tf.tanh(tf.nn.bias_add(tf.matmul(self.ob, W0), b0))

      self.pol_prob = tf.nn.softmax(tf.nn.bias_add(tf.matmul(
        h_1, W_softmax), b_softmax))
      self.val = tf.nn.bias_add(tf.matmul(h_1, W_linear), b_linear)

      actions_one_hot = tf.reshape(tf.one_hot(self.ac, output_size),
        (-1, output_size))
      masked_prob = tf.reduce_sum(actions_one_hot * self.pol_prob,
        reduction_indices=1)
      score = tf.mul(tf.log(tf.clip_by_value(
        masked_prob, 1.e-10, 1.0)), self.rew - self.val)
      value_loss = tf.nn.l2_loss(self.rew - self.val)

      self.pol_grads = tf.gradients(score, self.pol_vars)
      self.val_grads = tf.gradients(value_loss, self.val_vars)

      self.pol_updates = self.get_updates(global_network.pol_vars,
        self.pol_vars, self.pol_grads, global_network.pol_grad_msq)
      self.val_updates = self.get_updates(global_network.val_vars,
        self.val_vars, self.val_grads, global_network.val_grad_msq)

  def get_updates(self, global_vars, local_vars, grads, grad_msq,
                  momentum=0.9, epsilon=1e-9, grad_norm_clip=1.):
    updates = []
    for Wg, Wl, grad, msq in zip(global_vars, local_vars, grads, grad_msq):
      grad = tf.clip_by_norm(grad, grad_norm_clip)

      # compute rmsprop update per variable
      ms_update = momentum * msq + (1. - momentum) * tf.pow(grad, 2)
      gradient_update = -self.lr * grad / tf.sqrt(ms_update + epsilon)

      # apply updates to global variables, copy back to local variables
      l_to_g = Wg.assign_add(gradient_update)
      g_to_l = Wl.assign(Wg)

      updates += [l_to_g, g_to_l, ms_update]

    return updates

class ACAgent(object):
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
        episode_max_length = 100,
        num_threads = 1,
      )

    self.config.update(usercfg)

    with tf.variable_scope('global'):
      global_model = ThreadModel(self.nO, self.nA, None, self.config)

    self.thr_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.nO, self.nA, global_model, self.config)
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
          print('Thread: {} Steps: {} av100: {} lr: {}'.format(
            thread_id, env_steps, np.mean(ravg), lr))
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
        {model.ob:obs, model.ac:acts, model.rew:RR, model.lr:lr})

  def learn(self):
    ths = [threading.Thread(target=self.learning_thread, args=(i,)) for i in
      range(self.config['num_threads'])]

    for t in ths:
      t.start()

def main():
    env = gym.make("CartPole-v0")
    agent = ACAgent(env.observation_space, env.action_space,
      episode_max_length=10000, gamma=0.99, n_iter=1000000,
      num_threads=4, t_max=5, min_lr=0.0001, lr_decay_steps=30000)
    agent.learn()

if __name__ == "__main__":
    main()
