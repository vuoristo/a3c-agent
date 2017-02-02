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

Training the Agent
-----------------------
* Train the model with `python acagent.py`

Description
-----------
This project implements the Asynchronous Advantage Actor Critic reinforcement learning algorithm as described by DeepMind in their [A3C Paper](https://arxiv.org/abs/1602.01783v2). The policy network is implemented using TensorFlow and the training environments are provided by OpenAI Gym.

This implementation attemtps to follow the original specification closely while utilizing the powerful TensorFlow API.
