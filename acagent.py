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
    W0_size = config['hidden_1']
    rnn_size = config['rnn_size']
    num_rnn_cells = config['num_rnn_cells']
    num_rnn_steps = config['num_rnn_steps']
    entropy_beta = config['entropy_beta']
    grad_norm_clip_val = config['grad_norm_clip_val']

    self.ob = tf.placeholder(
      tf.float32, (None, num_rnn_steps, input_size), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')
    self.lr = tf.placeholder(tf.float32, name='lr')

    with tf.variable_scope('shared_variables') as sv_scope:
      init = tf.uniform_unit_scaling_initializer()
      self.W0 = tf.get_variable('W0', shape=(input_size, W0_size),
        initializer=init)
      self.b0 = tf.get_variable('b0', initializer=tf.constant(0.,
        shape=(W0_size,)))

      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_rnn_cells)

      obs_transposed = tf.transpose(self.ob, perm=(1,0,2))
      hidden_states = tf.map_fn(self._compute_hidden_state, obs_transposed)
      hs_transposed = tf.transpose(hidden_states, perm=(1,0,2))
      rnn_inputs = [tf.squeeze(hs, [1]) for hs in tf.split(1, num_rnn_steps,
        hs_transposed)]
      rnn_outputs, states = tf.nn.rnn(cell, rnn_inputs, dtype=tf.float32)

    with tf.variable_scope('policy') as pol_scope:
      W_softmax = tf.get_variable('W_softmax', shape=(rnn_size, output_size),
        initializer=init)
      b_softmax = tf.get_variable('b_softmax',
        initializer=tf.constant(0., shape=(output_size,)))

    with tf.variable_scope('value') as val_scope:
      W_linear = tf.get_variable('W_linear', shape=(rnn_size, 1),
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
      with tf.name_scope('outputs'):
        self.pol_prob = tf.nn.softmax(tf.nn.bias_add(tf.matmul(
          rnn_outputs[-1], W_softmax), b_softmax))
        self.val = tf.nn.bias_add(tf.matmul(rnn_outputs[-1], W_linear),
          b_linear)

      with tf.name_scope('targets'):
        actions_one_hot = tf.reshape(tf.one_hot(self.ac, output_size),
          (-1, output_size))
        masked_prob = tf.reduce_sum(actions_one_hot * self.pol_prob,
          reduction_indices=1)
        log_masked_prob = tf.log(tf.clip_by_value(masked_prob, 1.e-10, 1.0))
        entropy = -tf.reduce_sum(masked_prob * log_masked_prob)
        score = tf.reduce_sum(tf.transpose(tf.mul(log_masked_prob,
          tf.transpose(self.rew - self.val)))) + entropy * entropy_beta
        value_loss = 0.5 * tf.nn.l2_loss(self.rew - self.val)

      with tf.name_scope('gradients'):
        self.pol_grads = tf.gradients(score, self.pol_vars,
          name='gradients_pol')
        self.val_grads = tf.gradients(value_loss, self.val_vars,
          name='gradients_val')

      with tf.name_scope('updates'):
        self.pol_updates = self.get_updates(global_network.pol_vars,
          self.pol_vars, self.pol_grads, global_network.pol_grad_msq,
          grad_norm_clip=grad_norm_clip_val)
        self.val_updates = self.get_updates(global_network.val_vars,
          self.val_vars, self.val_grads, global_network.val_grad_msq,
          grad_norm_clip=grad_norm_clip_val)

  def _compute_hidden_state(self, observation):
    return tf.tanh(tf.nn.bias_add(tf.matmul(observation, self.W0), self.b0))

  def get_updates(self, global_vars, local_vars, grads, grad_msq,
                  momentum=0.99, epsilon=0.1,
                  grad_norm_clip=10.):
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
  def __init__(self, **usercfg):
    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.01,
        min_lr = 0.002,
        lr_decay_no_steps = 10000,
        hidden_1 = 20,
        rnn_size = 10,
        num_rnn_cells = 1,
        num_rnn_steps = 1,
        num_threads = 1,
        env_name = '',
        entropy_beta = 0.01,
        grad_norm_clip_val = 50.,
      )

    self.config.update(usercfg)

    self.envs = [gym.make(self.config['env_name']) for _ in
        range(self.config['num_threads'])]
    self.nO = self.envs[0].observation_space.shape[0]
    self.nA = self.envs[0].action_space.n

    with tf.variable_scope('global'):
      global_model = ThreadModel(self.nO, self.nA, None, self.config)

    self.thr_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.nO, self.nA, global_model, self.config)
        self.thr_models.append(thr_model)

    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def act(self, obs, model):
    w_size = self.config['num_rnn_steps']
    ob = np.reshape(obs[-w_size-1:-1], (1,w_size,-1))
    prob = self.sess.run(model.pol_prob, {model.ob:ob})
    action = categorical_sample(prob)
    return action

  def learning_thread(self, thread_id):
    t_max = self.config['t_max']
    lr = self.config['lr']
    min_lr = self.config['min_lr']
    lr_decay_no_steps = self.config['lr_decay_no_steps']
    lr_decay_step = (lr - min_lr)/lr_decay_no_steps
    w_size = self.config['num_rnn_steps']

    env = self.envs[thread_id]
    model = self.thr_models[thread_id]

    obs = deque(maxlen=t_max)
    acts = deque(maxlen=t_max)
    rews = deque(maxlen=t_max)

    ep_rews = 0
    ep_count = 0
    rews_acc = deque(maxlen=100)

    t = 0
    done = True
    for iteration in range(self.config['n_iter']):
      if done:
        done = False
        obs.clear()
        ob = env.reset()
        obs.extend([ob]*t_max)

        rews_acc.append(ep_rews)
        ep_rews = 0
        ep_count += 1
        print('Episode: {} Avg100: {}'.format(ep_count, np.mean(rews_acc)))
      else:
        action = self.act(list(obs), model)
        acts.append(action)
        ob, rew, done, _ = env.step(action)
        obs.append(ob)
        rews.append(rew)
        t += 1
        lr = lr - lr_decay_step if lr > min_lr else lr
        ep_rews += rew

      if done or t == t_max:
        obs_padded = np.r_[[obs[0]] * (w_size - 1), obs]
        obs_windowed = [obs_padded[i:i+w_size] for i in range(t)]
        obs_arr = np.reshape(obs_windowed, (t, w_size, self.nO))
        acts_arr = np.reshape(acts, (t, 1))

        last_ob = np.reshape(obs_arr[-1,:,:], (1, w_size, -1))
        R = 0 if done else self.sess.run(model.val, {model.ob:last_ob})
        R_arr = np.zeros((t, 1))
        R_arr[-1, 0] = R
        for i in reversed(range(t)):
          R = rews[i] + self.config['gamma'] * R
          R_arr[i, 0] = R

        _, _ = self.sess.run([model.pol_updates, model.val_updates],
          {model.ob:obs_arr, model.ac:acts_arr, model.rew:R_arr, model.lr:lr})

        acts.clear()
        rews.clear()
        t = 0

  def learn(self):
    ths = [threading.Thread(target=self.learning_thread, args=(i,)) for i in
      range(self.config['num_threads'])]

    for t in ths:
      t.start()

def main():
    agent = ACAgent(gamma=0.99, n_iter=100000, num_threads=4, t_max=5,
        min_lr=0.0001, lr_decay_no_steps=30000, num_rnn_cells=1,
        num_rnn_steps=4, env_name='CartPole-v0')
    agent.learn()

if __name__ == "__main__":
    main()
