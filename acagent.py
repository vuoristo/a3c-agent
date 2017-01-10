import gym
import tensorflow as tf
import numpy as np
import threading
from collections import deque
from PIL import Image, ImageOps

from util import _kernel_img_summary, _activation_summary, _input_summary

def weight_variable_conv(name, shape):
  return tf.get_variable(name,
    shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weight_variable(name, shape):
  return tf.get_variable(name,
    shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(name, shape, value):
  return tf.get_variable(name,
      shape=shape, initializer=tf.constant_initializer(value=value))

def conv2d(x, W, strides):
  return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

class ThreadModel(object):
  def __init__(self, input_shape, output_size, global_network, config):
    rnn_size = config['rnn_size']
    entropy_beta = config['entropy_beta']
    grad_norm_clip_val = config['grad_norm_clip_val']
    use_rnn = config['use_rnn']
    rms_decay = config['rms_decay']
    initial_lr = config['lr']
    initial_min_lr = config['min_lr']
    lr_decay_no_steps = config['lr_decay_no_steps']
    initial_lr_decay = (initial_lr - initial_min_lr)/lr_decay_no_steps

    self.ob = tf.placeholder(tf.float32, (None, *input_shape), name='ob')
    self.ac = tf.placeholder(tf.int32, (None, 1), name='ac')
    self.rew = tf.placeholder(tf.float32, (None, 1), name='rew')

    with tf.variable_scope('policy_value_network') as thread_scope:
      with tf.variable_scope('conv1'):
        W_conv1 = weight_variable_conv('W', [8,8,input_shape[2],16])
        b_conv1 = bias_variable('b', [16], 0)

        h_conv1 = tf.nn.relu(conv2d(self.ob, W_conv1, [1,4,4,1]) +
          b_conv1, name='h')

      with tf.variable_scope('conv2'):
        W_conv2 = weight_variable_conv('W', [4,4,16,32])
        b_conv2 = bias_variable('b', [32], 0)

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1,2,2,1]) +
          b_conv2, name='h')

      with tf.variable_scope('fc1'):
        conv2_out_size = 2592
        W_fc1 = weight_variable('W', [conv2_out_size, 256])
        b_fc1 = bias_variable('b', [256], 0.0)

        h_conv2_flat = tf.reshape(h_conv2, [-1, conv2_out_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name='h')

      nn_outputs = h_fc1
      nn_output_size = 256

      if use_rnn:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

        self.rnn_state_initial_c = tf.placeholder(tf.float32, (1, rnn_size))
        self.rnn_state_initial_h = tf.placeholder(tf.float32, (1, rnn_size))
        self.rnn_state_initial = tf.nn.rnn_cell.LSTMStateTuple(
          self.rnn_state_initial_c, self.rnn_state_initial_h)

        time_windows = tf.reshape(nn_outputs, (1, -1, nn_output_size))
        rnn_outputs, self.rnn_state_after = tf.nn.dynamic_rnn(
          lstm_cell, time_windows, initial_state=self.rnn_state_initial,
          dtype=tf.float32)
        nn_output_size = rnn_size
        nn_outputs = tf.reshape(rnn_outputs, (-1, rnn_size))

      W_softmax = weight_variable('W_softmax', [nn_output_size, output_size])
      b_softmax = bias_variable('b_softmax', [output_size], 0.0)

      W_linear = weight_variable('W_linear', [nn_output_size, 1])
      b_linear = bias_variable('b_linear', [1], 0.0)

    # Variable collections for update computations
    self.trainable_variables = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_scope.name)

    # Global model stores the RMSProp moving averages.
    # Thread models contain the evaluation and update logic.
    if global_network is None:
      self.gradient_mean_square = [tf.Variable(np.zeros(var.get_shape(),
        dtype=np.float32)) for var in self.trainable_variables]
      self.saver = tf.train.Saver(tf.global_variables())

      self.lr = tf.get_variable('lr_decay', initializer=tf.constant(initial_lr))
      self.lr_decay = tf.get_variable('lr', initializer=tf.constant(initial_lr_decay))
      self.min_lr = tf.get_variable('min_lr', initializer=tf.constant(initial_min_lr))
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
        log_masked_prob = tf.log(tf.clip_by_value(masked_prob, 1.e-22, 1.0))
        td_error = self.rew - tf.stop_gradient(self.val)
        policy_loss = -(log_masked_prob * td_error)
        value_loss = 0.5 * (self.rew - self.val) ** 2.
        total_loss = tf.reduce_sum(policy_loss + 0.5 * value_loss)

      with tf.name_scope('gradients'):
        self.grads = tf.gradients(total_loss, self.trainable_variables)

      with tf.name_scope('updates'):
        self.updates = self.get_rms_updates(global_network.trainable_variables,
          self.trainable_variables, self.grads,
          global_network.gradient_mean_square, global_network.lr, decay=rms_decay,
          grad_norm_clip=grad_norm_clip_val)

        lr_update = tf.cond(tf.greater(global_network.lr,
          global_network.min_lr),
          lambda: global_network.lr.assign(
            global_network.lr - global_network.lr_decay),
          lambda: global_network.lr)
        self.updates += [lr_update]

      # Get summaries for thread 0
      if 'thread_0' in thread_scope.name:
        with tf.variable_scope('summaries'):
          _kernel_img_summary(W_conv1, [8,8,1,16], 'conv1 kernels')
          _activation_summary(h_conv1, (20,20,16), 'conv1 activation')
          _input_summary(self.ob, (5,84,84,4), 'inputs')

          tf.summary.scalar('value_loss', tf.reduce_mean(value_loss))
          tf.summary.scalar('policy_loss', tf.reduce_mean(policy_loss))
          tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))
          tf.summary.scalar('max_prob', tf.reduce_max(self.pol_prob))
          tf.summary.scalar('td_error', tf.reduce_mean(td_error))
          tf.summary.scalar('learning_rate', global_network.lr)

          for grad in self.grads:
            summary_name = 'grad_' + '/'.join(grad.name.split('/')[-3:])
            tf.summary.scalar(summary_name, tf.reduce_mean(grad))

          for msq in global_network.gradient_mean_square:
            summary_name = 'msq_' + '/'.join(msq.name.split('/')[-3:])
            tf.summary.scalar(summary_name, tf.reduce_mean(msq))

          for var in self.trainable_variables:
            summary_name = 'var_' + '/'.join(var.name.split('/')[-3:])
            tf.summary.scalar(summary_name, tf.reduce_mean(var))

          self.summary_op = tf.summary.merge_all()

  def get_rms_updates(self, global_vars, local_vars, grads, grad_msq, lr,
                      decay=0.99, epsilon=1e-10, grad_norm_clip=50.):
    """
    Compute shared RMSProp updates for local_vars.
    global_vars - stores the global variables shared by all threads
    local_vars - thread local variables that are used for gradient computation
    grads - gradients of local_vars
    grad_msq - globally stored mean of squared gradients
    decay - the momentum parameter
    epsilon - for numerical stability
    grad_norm_clip - gradient norm clipping amount
    """
    updates = []
    for Wg, grad, msq in zip(global_vars, grads, grad_msq):
      grad = tf.clip_by_norm(grad, grad_norm_clip)
      msq_update = msq.assign(decay * msq + (1. - decay) * tf.pow(grad, 2))

      # control dependecies make sure msq is updated before gradients
      with tf.control_dependencies([msq_update]):
        gradient_update = -lr * grad / tf.sqrt(msq + epsilon)
        local_to_global = Wg.assign_add(gradient_update)

      updates += [gradient_update, local_to_global, msq_update]

    # control dependencies make sure local to global updates are completed
    # before global to local updates.
    with tf.control_dependencies(updates):
      for Wg, Wl in zip(global_vars, local_vars):
        global_to_local = Wl.assign(Wg)
        updates += [global_to_local]

    return updates

def resize_observation(observation, shape, centering):
  img = Image.fromarray(observation)
  img = ImageOps.fit(img, shape[:2], centering=centering)
  img = img.convert('L')
  return np.reshape(np.array(img), (1, *shape)) * 1./255.

def categorical_sample(prob):
  prob = np.asarray(prob)
  csprob = np.cumsum(prob)
  return (csprob > np.random.rand()).argmax()

def act(ob, model, session, use_rnn):
  if use_rnn:
    prob, running_rnn_state = session.run(
      [model.pol_prob, model.rnn_state_after],
      {model.ob:ob,
       model.rnn_state_initial_c:running_rnn_state[0],
       model.rnn_state_initial_h:running_rnn_state[1],
      })
  else:
    prob = session.run(model.pol_prob, {model.ob:ob})
  action = categorical_sample(prob)
  return action

def bootstrap_return(session, model, observation, running_rnn_state):
  if running_rnn_state is not None:
    R = session.run(
      model.val,
      {model.ob:observation,
       model.rnn_state_initial_c:running_rnn_state[0],
       model.rnn_state_initial_h:running_rnn_state[1],
      })
  else:
    R = session.run(model.val, {model.ob:observation})
  return R

def run_updates(session, model, obs_arr, acts_arr, R_arr, training_rnn_state):
  if training_rnn_state is not None:
    _, training_rnn_state = session.run(
      [model.updates, model.rnn_state_after],
      {model.ob:obs_arr,
       model.ac:acts_arr,
       model.rew:R_arr,
       model.rnn_state_initial_c:training_rnn_state[0],
       model.rnn_state_initial_h:training_rnn_state[1],
      })
  else:
    _ = session.run(
      [model.updates],
      {model.ob:obs_arr,
       model.ac:acts_arr,
       model.rew:R_arr,
      })
  return training_rnn_state

def reset_rnn_state(rnn_size):
  running_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
    np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
  training_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
    np.zeros([1, rnn_size]), np.zeros([1, rnn_size]))
  return running_rnn_state, training_rnn_state

def discounted_returns(rews, gamma, R, t):
  R_arr = np.zeros((t, 1))
  for i in reversed(range(t)):
    R = rews[i] + gamma * R
    R_arr[i, 0] = R
  return R_arr

def new_random_game(env, random_starts, action_size):
  ob = env.reset()
  no_rnd = np.random.randint(0, random_starts)
  for i in range(no_rnd):
    ob, _, _, _ = env.step(np.random.randint(action_size))
  return ob

def learning_thread(thread_id, config, session, model, global_model, env):
  t_max = config['t_max']
  use_rnn = config['use_rnn']
  rnn_size = config['rnn_size']
  window_size = config['window_size']
  gamma = config['gamma']
  random_starts = config['random_starts']

  ob_shape = (84,84,1)
  crop_centering = (0.5, 0.7)
  obs = np.zeros((t_max + window_size, *ob_shape))
  acts = np.zeros((t_max))
  rews = np.zeros((t_max))

  # variables for accounting
  ep_rews = 0
  ep_count = 0
  rews_acc = deque(maxlen=100)

  running_rnn_state = None
  training_rnn_state = None

  if thread_id == 0:
    summary_writer = tf.summary.FileWriter('train', session.graph)

  # Training loop
  t = 0
  done = True
  for iteration in range(config['n_iter']):
    if done:
      done = False
      ob = new_random_game(env, random_starts, env.action_space.n)
      ob = resize_observation(ob, ob_shape, crop_centering)
      obs[:] = ob

      if use_rnn:
        running_rnn_state, training_rnn_state = reset_rnn_state(rnn_size)

      rews_acc.append(ep_rews)
      ep_count += 1
      print('Thread: {} Episode: {} Rews: {} RunningAvgRew: '
            '{:.1f}'.format(thread_id, ep_count, ep_rews,
            np.mean(rews_acc)))
      ep_rews = 0
    else:
      obs_current_index = iteration % (t_max + window_size)
      obs[obs_current_index] = ob

      obs_start_index = obs_current_index - window_size
      obs_window_indices = np.arange(obs_start_index, obs_current_index)
      ob_w = obs[obs_window_indices]
      ob_w = np.transpose(ob_w, (3,1,2,0))

      action = act(ob_w, model, session, use_rnn)

      ob, rew, done, _ = env.step(action)
      ob = resize_observation(ob, ob_shape, crop_centering)
      acts[t] = action
      rews[t] = rew if not done else 0
      t += 1
      ep_rews += rew

    if done or t == t_max:
      obs_current_index = iteration % (t_max + window_size)
      obs_start_index = obs_current_index - t
      obs_indices = [np.arange(i - window_size + 1, i + 1) for i in np.arange(obs_start_index, obs_current_index)]
      obs_indices = np.reshape(obs_indices, (t, window_size)) % (t_max + window_size)
      obs_arr = obs[obs_indices]
      obs_arr = np.transpose(np.reshape(obs_arr, (t,window_size,84,84)), (0,2,3,1))

      acts_arr = np.reshape(acts[:t], (t, 1))

      if done:
        R = 0
      else:
        R = bootstrap_return(session, model, ob_w, running_rnn_state)

      R_arr = discounted_returns(rews, gamma, R, t)

      training_rnn_state = run_updates(session, model, obs_arr, acts_arr,
        R_arr, training_rnn_state)

      if thread_id == 0 and iteration % 100 == 0:
        summary_str = session.run(model.summary_op,
          {model.ob:obs_arr,
           model.ac:acts_arr,
           model.rew:R_arr})
        summary_writer.add_summary(summary_str, iteration)

      if thread_id == 0 and ep_count % 300 == 0:
        global_model.saver.save(session, 'train/model.ckpt',
          global_step=iteration)

      acts[:] = 0
      rews[:] = 0
      t = 0

class ACAgent(object):
  def __init__(self, **usercfg):
    self.config = dict(
        n_iter = 100,
        t_max = 10,
        gamma = 0.98,
        lr = 0.01,
        min_lr = 0.002,
        lr_decay_no_steps = 10000,
        rnn_size = 10,
        use_rnn = True,
        num_threads = 1,
        window_size = 4,
        env_name = '',
        entropy_beta = 0.01,
        grad_norm_clip_val = 50.,
        rms_decay = 0.99,
        random_starts=30,
      )

    self.config.update(usercfg)

    self.envs = [gym.make(self.config['env_name']) for _ in
      range(self.config['num_threads'])]

    self.input_shape = (84,84,4)
    self.action_num = self.envs[0].action_space.n
    self.use_rnn = self.config['use_rnn']

    with tf.variable_scope('global'):
      self.global_model = ThreadModel(self.input_shape, self.action_num, None,
        self.config)

    self.thread_models = []
    for thr in range(self.config['num_threads']):
      with tf.variable_scope('thread_{}'.format(thr)):
        thr_model = ThreadModel(self.input_shape, self.action_num,
          self.global_model, self.config)
        self.thread_models.append(thr_model)

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def learn(self):
    threads = []
    for i in range(self.config['num_threads']):
      threads.append(threading.Thread(target=learning_thread, args=(
        i, self.config, self.session, self.thread_models[i], self.global_model,
        self.envs[i])))

    self.session.graph.finalize()

    for thread in threads:
      thread.start()

def main():
  agent = ACAgent(gamma=0.99, n_iter=10000000, num_threads=1, t_max=5, lr=0.001,
    min_lr=0.000001, lr_decay_no_steps=10000000, rnn_size=100,
    env_name='Breakout-v0', use_rnn=False, entropy_beta=0.01, rms_decay=0.99)
  agent.learn()

if __name__ == "__main__":
  main()
