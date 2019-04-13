# Project 1: Navigation

### Introduction

This project trains an agent to solve the Unity "Reacher" environment.  In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment includes 20 simultaneous agents to enable faster training.

### Dependencies

Directions for installing dependencies can be found at:
https://github.com/udacity/deep-reinforcement-learning#dependencies

You'll also need to clone the above project, change into the python directory,
and install the dependencies:

`pip install -e .`

### Getting Started

1. Download the environment:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Instructions

To train the agent, you should run main.py.  For example:

`pythonw main.py --episodes 200 --max-steps 1000 --saveto checkpoint.pth --saveplot scores.png --environnment path/to/Reacher.app`

To run in eval mode:

`pythonw main.py --eval=True --loadfrom checkpoint.pth --episodes 10`
