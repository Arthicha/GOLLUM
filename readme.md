# GOLLUM: Growable Online Locomotion Learning Under Multicondition

The videos are available at 
- [continual locomotion learning on different slopes](https://youtu.be/znNi1mlLjEQ)
- [continual locomotion learning on different slopes with possible motor dysfunction](https://youtu.be/HugIBO6cnNo)
- [continual locomotion learning on different terrains](https://youtu.be/fGCy8CXPuO0) 

<p align="center">
    <img width="100%" src="/pictures/locomotion_terrain.JPG" />
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

| Location | Parameter | Component | Meaning  |
| ------------- | ------------- | ------------- | ------------- |
| network.ini | W_TIME | neural control | transition speed/walking freqeuncy | 
| mn_optim.ini | MINGRAD | motor mapping | gradient clipping (prevent exploding gradient) | 
|  | LR | motor mapping |learning rate | 
|  | SIGMA | motor mapping | starting exploration standard deviation (between SIGMAMIN and SIGMAMAX)|
|  | SIGMAMIN | motor mapping | minimum exploration standard deviation|
|  | SIGMAMAX | motor mapping | maximum exploration standard deviation|
| pmn_optim.ini | MINGRAD | premotor mapping | gradient clipping (prevent exploding gradient) | 
|  | LR | premotor mapping |learning rate | 
|  | SIGMA | premotor mapping | starting exploration standard deviation (between SIGMAMIN and SIGMAMAX)|
|  | SIGMAMIN | premotor mapping | minimum exploration standard deviation|
|  | SIGMAMAX | premotor mapping | maximum exploration standard deviation|
| vn_optim.ini | LR | value prediction | learning rate |
|  | ITERATION | value prediction | number of repeat update iteration/epoch | 
| on_optim.ini | LR | observation prediction | learning rate |
|  | ITERATION | observation prediction | number of repeat update iteration/epoch | 
| vbn_optim.ini | LR | value uncertainty prediction | learning rate |
|  | ITERATION | value uncertainty prediction | number of repeat update iteration/epoch | 
| obn_optim.ini | LR | observation uncertainty prediction | learning rate |
|  | ITERATION | observation uncertainty prediction | number of repeat update iteration/epoch | 
| main.py | NREPLAY | training process | number of episodes/roll-outs used |
|  | NTIMESTEP | training process  | number of timesteps per episode | 
|  | NEPISODE | training process  | number of episode used for learning | 
|  | RESET | training process  | enable simulation/network reset | 
|  |  | | (reset the simulation and the network after each episode ends) | 

4. Enjoy! With a proper set of hyperparameters, the robot should start walking (forward/backward) within the first 40 episodes after switching the target, while successfully recall previous skills when switching back to the other target.

