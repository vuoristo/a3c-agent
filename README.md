a3c-agent
=========

Requirements
------------
* pip
* virtualenv (recommended)
* python 3
* numpy
* tensorflow 0.12
* OpenAI Gym with Atari environments

Installation
------------
* Clone this repository
* Create new virtualenv environment
* For Mac: Install requirements with `pip install -r requirements.txt`
* For other OSs edit requirements.txt with correct tensorflow version

Running the Training / Evaluation
---------------------------------
* Train the model with `python AtariAgent.py --env <env_name>`
* Evaluate a trained model with `python AtariAgent.py --env <env_name> --load_weights <path_to_weights> --evaluate --render`

Description
-----------
This project implements the Asynchronous Advantage Actor Critic reinforcement learning algorithm as described by DeepMind in their [A3C Paper](https://arxiv.org/abs/1602.01783v2). The policy network is implemented using TensorFlow and the training environments are provided by OpenAI Gym.

This implementation attemtps to follow the original specification closely while utilizing the powerful TensorFlow API.

TODOs
-----
* Cleaner thread handling. Utilize TF queues, monitors and runners.
* Evaluate on multiple environments. Provide pretrained weights.
