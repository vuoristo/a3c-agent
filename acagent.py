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
    entropy_beta = config['entropy_beta']
    grad_norm_clip_val = config['grad_norm_clip_val']
    use_rnn = config['use_rnn']
    rms_decay = config['rms_decay']

    self.ob = tf.placeholder(tf.float32, (None, input_size), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')
    self.lr = tf.placeholder(tf.float32, name='lr')

    with tf.variable_scope('policy_value_network') as thread_scope:
      init = tf.uniform_unit_scaling_initializer()
      W0 = tf.get_variable('W0', shape=(input_size, W0_size), initializer=init)
      b0 = tf.get_variable('b0', initializer=tf.constant(0., shape=(W0_size,)))
      W1 = tf.get_variable('W1', shape=(W0_size, W0_size), initializer=init)
      b1 = tf.get_variable('b1', initializer=tf.constant(0., shape=(W0_size,)))
      h0 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.ob, W0), b0))
      h1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h0, W1), b1))

      nn_output_size = W0_size
      nn_outputs = h1

      if use_rnn:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

        self.rnn_state_initial_c = tf.placeholder(tf.float32, (1, rnn_size))
        self.rnn_state_initial_h = tf.placeholder(tf.float32, (1, rnn_size))
        self.rnn_state_initial = tf.nn.rnn_cell.LSTMStateTuple(
          self.rnn_state_initial_c, self.rnn_state_initial_h)

        time_windows = tf.reshape(h1, (1, -1, W0_size))
        rnn_outputs, self.rnn_state_after = tf.nn.dynamic_rnn(
          lstm_cell, time_windows, initial_state=self.rnn_state_initial,
          dtype=tf.float32)
        nn_output_size = rnn_size
        nn_outputs = tf.reshape(rnn_outputs, (-1, rnn_size))

      W_softmax = tf.get_variable('W_softmax', shape=(nn_output_size,
        output_size), initializer=init)
      b_softmax = tf.get_variable('b_softmax',
        initializer=tf.constant(0., shape=(output_size,)))

      W_linear = tf.get_variable('W_linear', shape=(nn_output_size, 1),
        initializer=init)
      b_linear = tf.get_variable('b_linear',
        initializer=tf.constant(0., shape=(1,)))

    # Variable collections for update computations
    self.trainable_variables = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_scope.name)

    # Global model stores the RMSProp moving averages.
    # Thread models contain the evaluation and update logic.
    if global_network is None:
      self.gradient_mean_square = [tf.Variable(np.zeros(var.get_shape(),
        dtype=np.float32)) for var in self.trainable_variables]
    else:
      with tf.name_scope('outputs'):
        self.pol_prob = tf.nn.softmax(tf.nn.bias_add(tf.matmul(
          nn_outputs, W_softmax), b_softmax))
        self.val = tf.nn.bias_add(tf.matmul(nn_outputs, W_linear),
          b_linear)

      with tf.name_scope('targets'):
        actions_one_hot = tf.reshape(tf.one_hot(self.ac, output_size),
          (-1, output_size))
        masked_prob = tf.reduce_sum(actions_one_hot * self.pol_prob,
          reduction_indices=1, keep_dims=True)
        log_masked_prob = tf.log(tf.clip_by_value(masked_prob, 1.e-10, 1.0))
        td_error = self.rew - self.val
        policy_loss = -log_masked_prob * td_error
        value_loss = tf.nn.l2_loss(td_error)
        total_loss = tf.reduce_mean(policy_loss + 0.5 * value_loss)

      with tf.name_scope('gradients'):
        self.grads = tf.gradients(total_loss, self.trainable_variables)

      with tf.name_scope('updates'):
        self.updates = self.get_updates(global_network.trainable_variables,
          self.trainable_variables, self.grads,
          global_network.gradient_mean_square, decay=rms_decay)

  def get_updates(self, global_vars, local_vars, grads, grad_msq,
                  decay=0.9, epsilon=1e-10, grad_norm_clip=10.):
    updates = []
    for Wg, grad, msq in zip(global_vars, grads, grad_msq):
      # compute rmsprop update per variable
      msq_update = msq.assign(decay * msq + (1. - decay) * tf.pow(grad, 2))
      with tf.control_dependencies([msq_update]):
        gradient_update = -self.lr * grad / tf.sqrt(msq + epsilon)
        l_to_g = Wg.assign_add(gradient_update)

      # apply updates to global variables
      updates += [gradient_update, l_to_g, msq_update]

    with tf.control_dependencies(updates):
      for Wg, Wl in zip(global_vars, local_vars):
        g_to_l = Wl.assign(Wg)
        updates += [g_to_l]

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
        use_rnn = True,
        num_threads = 1,
        env_name = '',
        entropy_beta = 0.01,
        grad_norm_clip_val = 100.,
        rms_decay = 0.99,
      )

    self.config.update(usercfg)

    self.envs = [gym.make(self.config['env_name']) for _ in
      range(self.config['num_threads'])]
    self.nO = self.envs[0].observation_space.shape[0]
    self.nA = self.envs[0].action_space.n
    self.use_rnn = self.config['use_rnn']

    with tf.variable_scope('global'):
      global_model = ThreadModel(self.nO, self.nA, None, self.config)

    self.thr_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.nO, self.nA, global_model, self.config)
        self.thr_models.append(thr_model)

    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def reset_rnn_state(self, rnn_size):
    self.running_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
      np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
    self.training_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
      np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))

  def act(self, ob, model):
    if self.use_rnn:
      prob, self.running_rnn_state = self.sess.run(
        [model.pol_prob, model.rnn_state_after],
        {model.ob:ob,
         model.rnn_state_initial_c:self.running_rnn_state[0],
         model.rnn_state_initial_h:self.running_rnn_state[1],
        })
    else:
      prob = self.sess.run(model.pol_prob, {model.ob:ob})
    action = categorical_sample(prob)
    return action

  def learning_thread(self, thread_id):
    t_max = self.config['t_max']
    lr = self.config['lr']
    min_lr = self.config['min_lr']
    lr_decay_no_steps = self.config['lr_decay_no_steps']
    lr_decay_step = (lr - min_lr)/lr_decay_no_steps
    self.use_rnn = self.config['use_rnn']
    rnn_size = self.config['rnn_size']

    env = self.envs[thread_id]
    model = self.thr_models[thread_id]

    ob_shape = env.observation_space.shape
    obs = np.zeros((t_max, *ob_shape))
    acts = np.zeros((t_max))
    rews = np.zeros((t_max))

    ep_rews = 0
    ep_count = 0
    rews_acc = deque(maxlen=100)

    if self.use_rnn:
      self.reset_rnn_state(rnn_size)

    t = 0
    done = True
    for iteration in range(self.config['n_iter']):
      if done:
        if self.use_rnn:
          self.reset_rnn_state(rnn_size)
        done = False
        ob = env.reset()
        ob = np.reshape(ob, (1, *ob_shape))
        rews_acc.append(ep_rews)
        ep_count += 1
        print('Thread: {} Episode: {} Rews: {} RunningAvgRew: '
              '{:.1f} lr: {}'.format(thread_id, ep_count, ep_rews,
              np.mean(rews_acc), lr))
        ep_rews = 0
      else:
        obs[t] = ob
        action = self.act(ob, model)
        ob, rew, done, _ = env.step(action)
        ob = np.reshape(ob, (1, *ob_shape))
        acts[t] = action
        rews[t] = rew if not done else 0
        t += 1
        ep_rews += rew

      if done or t == t_max:
        obs_arr = np.reshape(obs[:t], (t, *ob_shape))
        acts_arr = np.reshape(acts[:t], (t, 1))

        if done:
          R = 0
        else:
          if self.use_rnn:
            R = self.sess.run(
              model.val,
              {model.ob:ob,
               model.rnn_state_initial_c:self.running_rnn_state[0],
               model.rnn_state_initial_h:self.running_rnn_state[1],
              })
          else:
            R = self.sess.run(model.val, {model.ob:ob})

        R_arr = np.zeros((t, 1))
        for i in reversed(range(t)):
          R = rews[i] + self.config['gamma'] * R
          R_arr[i, 0] = R

        if self.use_rnn:
          _, self.training_rnn_state = self.sess.run(
            [model.updates, model.rnn_state_after],
            {model.ob:obs_arr,
             model.ac:acts_arr,
             model.rew:R_arr,
             model.lr:lr,
             model.rnn_state_initial_c:self.training_rnn_state[0],
             model.rnn_state_initial_h:self.training_rnn_state[1],
            })
        else:
          _ = self.sess.run(
            model.updates,
            {model.ob:obs_arr,
             model.ac:acts_arr,
             model.rew:R_arr,
             model.lr:lr,
            })

        lr = lr - lr_decay_step if lr > min_lr else lr

        acts[:] = 0
        rews[:] = 0
        obs[:] = 0
        t = 0

  def learn(self):
    ths = [threading.Thread(target=self.learning_thread, args=(i,)) for i in
      range(self.config['num_threads'])]

    for t in ths:
      t.start()

def main():
    agent = ACAgent(gamma=0.99, n_iter=10000000, num_threads=1, t_max=5,
      lr=0.01, min_lr=0.1, lr_decay_no_steps=30000, hidden_1=100, rnn_size=100,
      env_name='CartPole-v0', use_rnn=False, entropy_beta=0.0,
      rms_decay=0.9)
    agent.learn()

if __name__ == "__main__":
    main()
