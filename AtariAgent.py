import argparse

from acagent import ACAgent

ENV_NAME = 'Breakout-v0'
def main():
  parser = argparse.ArgumentParser('Train or Evaluate a DQN Agent for OpenAI '
      'Gym Atari Environments')
  parser.add_argument('--env', '-e', default=ENV_NAME)
  parser.add_argument('--evaluate', action='store_true', default=False)
  parser.add_argument('--load_weights', '-l', default=None)
  parser.add_argument('--render', '-r', action='store_true', default=False)

  args = parser.parse_args()
  env_name = args.env
  weights_to_load = args.load_weights
  evaluate = args.evaluate
  render = args.render

  agent = ACAgent(gamma=0.99, n_iter=10000000, num_threads=1, t_max=5,
    lr=0.001, min_lr=0.000001, lr_decay_no_steps=10000000, rnn_size=256,
    window_size=1, env_name=env_name, use_rnn=True, entropy_beta=0.01,
    rms_decay=0.99, load_path=weights_to_load, evaluate=evaluate,
    render=render)
  agent.learn()

if __name__ == '__main__':
  main()
