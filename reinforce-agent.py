import gym
import tensorflow as tf
import numpy as np

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x), 'float64')
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()

def ml_sample(prob_n):
    prob_n = np.asarray(prob_n)
    return prob_n.argmax()

def get_traj(agent, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    ob = env.reset()
    obs = []
    acts = []
    rews = []
    for _ in range(episode_max_length):
        if render:
          a = agent.act2(ob)
        else:
          a = agent.act(ob)
        (ob, rew, done, _) = env.step(a)
        obs.append(ob)
        acts.append(a)
        rews.append(rew)
        if done: break
        if render: env.render()
    return {"reward" : np.array(rews),
            "ob" : np.array(obs),
            "action" : np.array(acts)
            }

class REINFORCEAgent(object):
  def __init__(self, obs_space, action_space, **usercfg):
    nO = obs_space.shape[0]
    nA = action_space.n

    self.config = dict(
        ts_per_batch = 10000,
        n_iter = 100,
        gamma = 0.98,
        stepsize = 0.05,
        hidden_size = 20,
        episode_max_length = 100,
      )

    self.config.update(usercfg)

    self.ob_no = tf.placeholder(tf.float32, (None, nO))
    self.a_n = tf.placeholder(tf.int32, (None))
    self.adv_n = tf.placeholder(tf.float32, (None))

    hs = self.config['hidden_size']
    init = tf.constant(np.random.randn(nO, hs)/np.sqrt(nO), dtype=tf.float32)
    W0 = tf.get_variable('W0',initializer=init)
    b0 = tf.Variable(tf.constant(0., shape=(hs,)))
    init2 = tf.constant(1e-4*np.random.randn(hs, nA), dtype=tf.float32)
    W1 = tf.get_variable('W1',initializer=init2)
    b1 = tf.Variable(tf.constant(0., shape=(nA,)))

    prob_na = tf.nn.softmax(tf.matmul(tf.tanh(tf.matmul(self.ob_no, W0) + b0), W1) + b1)

    a_n_oh = tf.reshape(tf.one_hot(self.a_n, nA), (-1, nA))
    masked_prob_na = tf.reduce_sum(a_n_oh * prob_na, reduction_indices=1)
    loss = -tf.reduce_mean(tf.mul(tf.log(masked_prob_na), self.adv_n))

    stepsize = self.config['stepsize']
    optimizer = tf.train.RMSPropOptimizer(stepsize, momentum=0.9, epsilon=1e-9)
    self.train = optimizer.minimize(loss)
    self.compute_prob = prob_na
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

    self.dbg_var = a_n_oh * prob_na

  def act(self, ob):
    prob = self.sess.run(self.compute_prob, {self.ob_no:np.reshape(ob, (1, -1))})
    action = categorical_sample(prob)
    return action

  def act2(self, ob):
    prob = self.sess.run(self.compute_prob, {self.ob_no:np.reshape(ob, (1, -1))})
    action = ml_sample(prob)
    return action

  def learn(self, env):
    for iteration in range(self.config['n_iter']):
      trajs = []
      timesteps_total = 0
      while timesteps_total < self.config['ts_per_batch']:
        traj = get_traj(self, env, self.config['episode_max_length'])
        trajs.append(traj)
        timesteps_total += len(traj['reward'])

      all_ob = np.concatenate([traj['ob'] for traj in trajs])
      rets = [discount(traj['reward'], self.config['gamma']) for traj in trajs]
      maxlen = max(len(ret) for ret in rets)
      padded_rets = [np.concatenate([ret, np.zeros(maxlen-len(ret))]) for ret in rets]
      baseline = np.mean(padded_rets, axis=0)
      advs = [ret - baseline[:len(ret)] for ret in rets]
      all_action = np.concatenate([traj['action'] for traj in trajs])
      all_adv = np.concatenate(advs)

      dbg, _ = self.sess.run([self.dbg_var, self.train], {self.ob_no:all_ob, self.a_n:all_action, self.adv_n:all_adv})
      np.set_printoptions(threshold=np.inf)
      #print(dbg)
      eprews = np.array([traj["reward"].sum() for traj in trajs]) # episode total rewards
      eplens = np.array([len(traj["reward"]) for traj in trajs]) # episode lengths
      # Print stats
      print("-----------------")
      print("Iteration: {}".format(iteration))
      print("NumTrajs: {}".format(len(eprews)))
      print("NumTimesteps: {}".format(np.sum(eplens)))
      print("MaxRew: {}".format(eprews.max()))
      print("MeanRew: {} +- {}".format(eprews.mean(), eprews.std()/np.sqrt(len(eprews))))
      print("MeanLen: {} +- {}".format(eplens.mean(), eplens.std()/np.sqrt(len(eplens))))
      print("-----------------")
      get_traj(self, env, self.config["episode_max_length"], render=False)

def main():
    env = gym.make("CartPole-v0")
    agent = REINFORCEAgent(env.observation_space, env.action_space,
        episode_max_length=300, stepsize=0.01, hidden_size=30, gamma=0.999)
    agent.learn(env)

if __name__ == "__main__":
    main()
