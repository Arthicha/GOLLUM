# GOLLUM: Growable Online Locomotion Learning Under Multicondition

The videos are available at 
- [continual locomotion learning on different slopes](https://youtu.be/znNi1mlLjEQ)
- [continual locomotion learning on different slopes with possible motor dysfunction](https://youtu.be/HugIBO6cnNo)
- [continual locomotion learning on different terrains](https://youtu.be/fGCy8CXPuO0) 

<p align="center">
    <img width="75%" src="/pictures/locomotion_terrain.JPG" />
</p>

# Contents
- [Requirements](#Requirements)
- [Demonstration](#Demonstration)
- [Running](#Running)

# Requirements

* simulation software
	- CoppeliaSim 4.4.0 (at least)
	- Mujoco physic engine (come with Coppeliasim > 4.4.0)

* python 3.6.5
	- numpy 1.19.5 (at least)
	- pytorch 1.5.0+cu92 (at least)

# Demonstration

The simulation-based demonstration is provided as a fundamental proof of concepts of GOLLUM. The objective of the robot is to walk forward or backward (according to the target direction command) with the maximum speed. The robot receives one input: the target direction command, which modifies the reward received from the interaction with the environment. In this setup, the target direction command was switch between +1.0 and -1.0 every 100 learning episodes. Note that, the robot behavior for locomotion learning (in the simulation) is not constrained by any means, so the robot will do what ever it takes to get the maximum reward. 

<p align="center">
    <img width="75%" src="/pictures/simulation.PNG" />
</p>
 
# Running

1. Open the CoppeliaSim scene locating at `simulation/MORF_ContinualLocomotionLearning`

2. In order to start the training, just run the following command:

```
python main.py
```

3. If you want to try different hyperparameter values, you can modify them according to the table below.

| Location | Parameter | Meaning  |
| ------------- | ------------- | ------------- |
| network.ini | W_TIME | transition speed/walking freqeuncy | 
| optimizer.ini | MINGRAD | gradient clipping (prevent exploding gradient) | 
|  | LR | learning rate | 
|  | SIGMA | starting exploration standard deviation (between 0.001-0.05)|
| main.py | NREPLAY | number of episodes/roll-outs used |
|  | NTIMESTEP | number of timesteps per episode | 
|  | NEPISODE | number of episode used for learning | 
|  | RESET | enable simulation/network reset | 
|  |  | (reset the simulation and the network after each episode ends) | 

4. Enjoy! With a proper set of hyperparameters, the robot should start walking (forward/backward) within the first 40 episodes after switching the target, while successfully recall previous skills when switching back to the other target.

