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
        lr_decay_no_steps = 10000,
        hidden_1 = 20,
        num_rnn_cells = 16,
        num_rnn_steps = 2,
        num_threads = 1,
        env_name = '',
      )

    self.config.update(usercfg)

    with tf.variable_scope('global'):
      global_model = ThreadModel(self.nO, self.nA, None, self.config)

    self.thr_models = []
    self.envs = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.nO, self.nA, global_model, self.config)
        self.thr_models.append(thr_model)
      self.envs.append(gym.make(self.config['env_name']))

    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def act(self, ob, model):
    prob = self.sess.run(model.pol_prob, {model.ob:np.reshape(ob, (1, -1))})
    action = categorical_sample(prob)
    return action

  def learning_thread(self, thread_id):
    t_max = self.config['t_max']
    lr = self.config['lr']
    min_lr = self.config['min_lr']
    lr_decay_no_steps = self.config['lr_decay_no_steps']
    lr_decay_step = (lr - min_lr)/lr_decay_no_steps
    num_rnn_steps = self.config['num_rnn_steps']

    env = self.envs[thread_id]
    model = self.thr_models[thread_id]

    obs = deque(maxlen=t_max)
    acts = deque(maxlen=t_max)
    rews = deque(maxlen=t_max)

    t = 0
    done = True
    for iteration in range(self.config['n_iter']):
      if done:
        t = 0
        done = False
        obs.clear()
        ob = env.reset()
        obs.extend([ob]*t_max)
      else:
        action = self.act(ob, model)
        acts.append(action)
        ob, rew, done, _ = env.step(action)
        obs.append(ob)
        rews.append(rew)
        t += 1
        lr = lr - lr_decay_step if lr > min_lr else lr

      if done or t == t_max:
        obs_padded = np.r_[[obs[0]] * (num_rnn_steps - 1), obs]
        obs_windowed = [obs_padded[i:i+num_rnn_steps] for i in range(t)]
        obs_arr = np.reshape(obs_windowed, (t, num_rnn_steps, self.nO))
        acts_arr = np.reshape(acts, (t, 1))

        R = 0 if done else self.sess.run(model.val, {model.ob:obs_arr[-1,:,:]})
        R_arr = np.zeros((t, 1))
        R_arr[-1, 0] = R
        for i in reversed(range(t)):
          R = rews[i] + self.config['gamma'] * R
          R_arr[i, 0] = R

        _, _ = self.sess.run([model.pol_updates, model.val_updates],
          {model.ob:obs_arr, model.ac:acts_arr, model.rew:R_arr, model.lr:lr})

        acts.clear()
        rews.clear()

  def learn(self):
    ths = [threading.Thread(target=self.learning_thread, args=(i,)) for i in
      range(self.config['num_threads'])]

    for t in ths:
      t.start()

def main():
    env = gym.make("CartPole-v0")
    agent = ACAgent(env.observation_space, env.action_space, gamma=0.99,
      n_iter=10000, num_threads=1, t_max=5, min_lr=0.0001,
      lr_decay_no_steps=30000, env_name='CartPole-v0')
    agent.learn()

if __name__ == "__main__":
    main()
