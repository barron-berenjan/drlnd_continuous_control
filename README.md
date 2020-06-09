# Continuous Control with Twin Delayed DDPG

This repository provides the code required to train an agent using **Twin Delayed DDPG (TD3)** to solve the _Continuous Control_ assignment from Udacity's Deep Reinforcement Learning Nanodgree

## Environment

In this project, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal is to train an agent that maintains its position at the target location for as many time steps as possible. 

The __state space__ has 33 dimensions corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. include the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

The task is episodic, and to solve the environment, the agent requires to get an average score of +30 over 100 consecutive episodes.

## Dependencies and Set-up

To set up your python environment to run the code in this repository, follow the instructions below.

1. Clone this repository

2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
  
 3. Clone the [DRLND repository](https://github.com/udacity/deep-reinforcement-learning), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

 4. The environment for this project is based on the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). For this project you will __not need__ to install Unity and a pre-built environment can be downloaded from one of the links below. You need only select the environment that matches your operating system:

	- __Linux__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
	- __Mac OSX__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	- __Windows (32-bit)__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	- __Windows (64-bit)__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

__Note: Unzip the file in the same directory as the notebooks from this repository.__


## Training the Agent

Run the cells in `Continuous Control with TD3 (Train).ipynb` to train the agent. The agent will stop training once it reaches an average score of +30 over 100 consecutive episodes

## Watch a Trained Agent in Action

Run `Continuous Control with TD3 (Test).ipynb` to watch a pre-trained agent in action!



