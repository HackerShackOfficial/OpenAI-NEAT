# OpenAI-Neat

Watch the original video [here](https://youtu.be/o1_SkiEAjmA)

## Gym Setup Guide

### Installation Guide
First, make sure to follow the install instructions for OpenAI Gym at this [link](https://gym.openai.com/docs).

You'll need to have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [pip](https://pip.pypa.io/en/stable/installing/) installed on your machine.

Next, make sure to install the proper python packages for this project.
```shell
pip install scipy neat-python==0.8 argparse
```

### Running the Program

There are two files used to solve the gym environment: `gym_config` and `gym_solver.py`. `gym_config` contains the config for the neuroevolution process and `gym_solver.py` is the program the creates the neural networks and solves the game. You will need to adjust parameters in both to solve a game. 

If you open the gym_solver.py file with an IDE like [pycharm](https://www.jetbrains.com/pycharm/), there will be a section at the top for user parameters. 

```python
### User Params ###

# The name of the game to solve
game_name = 'CartPole-v0'

### End User Params ###
```

There will be a variable for a game name. This can be any game on the [OpenAI Gym website](https://gym.openai.com/envs). For now, keep CartPole since it is one of the most basic games.

To run the program, navigate to the project's directory in your terminal. This will be the same folder you cloned into (OpenAI-Neat). There are several different parameters that you can run the program with.

`--max-steps`: The max number of steps to take per genome (timeout)

`--episodes`: The number of times to run a single genome. This takes the average fitness score over all episodes for one genome

`--render`: Renders the game while the algorithm is learning

`--generations`: The number of generations to evolve the network

`--checkpoint`: Uses a checkpoint to start the simulation

`--num-cores`: The number cores on your computer for parallel execution (not in `--render` mode)

To run the simulation, execute this command:

```shell
python gym_solver.py --max-steps=1000 --episodes=10 --generations=50 --render
```
This tells the program to run 50 generations with 10 episodes per species in the population and render the game while the algorithm is learning. 

### Editing Parameters
If you want to change the game, you will need to edit a few parameters. As an example, let's say we want to play Pacman. In order to play Atari games, we must use the ram version. Currently only ram versions are compatible with my program. On the Atari [page](https://gym.openai.com/envs#atari), scroll down and find the name of the ram version of Pacman (`MsPacman-ram-v0`).

Open the `gym_solver.py` file and edit the `game_name` parameter so that it is `'MsPacman-ram-v0'`. 

Next, you will need to edit the `gym_config` file. However, before you do that, you will need to run the `gym_solver.py` program once to find some useful information.

```shell
python gym_solver.py
```

The program will fail to run, but that is okay. We need two values before we update the config. 
There will be two lines at the start of our program output:

```
Input Nodes: 128
Output Nodes: 9
```

Use these values to adjust the values for the following in the `gym_config` file:

```
input_nodes          = 128
output_nodes         = 9
```

You will also need to decide a target high score. For Pacman, I chose `6000`. You may have to run the simulation to decide a proper score. This value will also need to be updated in your `gym_config` file:

```
max_fitness_threshold   = 6000
```

Finally we can run our program. In this case, `--max-steps` will be our timeout. I suggest setting a high value like `10000` since it will take a while to accumulate a score. For this simulation, we won't render the game because it simulates much faster. We can also take advantage of multiple processes. Since my computer has 8 cores, I set `--num-cores` to `8`

```
python gym_solver.py --episodes=3 --generations=100 --num-cores=8 --max-steps=10000
```

It will take a few hours to simulate this game.

### Starting from a Checkpoint

My program also gives you the ability to continue a simulation after it finishes. When your simulation finishes, it will generate a `checkpoint` file. If you start a new simulation on the same game, you can use this checkpoint file to pick up where your simulation left off. In the Pacman example, after my simulation finishes, I can run:

```
python gym_solver.py --episodes=3 --generations=100 --num-cores=8 --max-steps=10000 --checkpoint=checkpoint
```
to simulate another 100 generations. At the end of this simulation, another checkpoint file will be generated. 

## Universe Setup Guide

First, make sure to follow the installation guide for gym (above).

Then, make sure you follow the installation instructions for universe at this [link](https://github.com/openai/universe#installation).

### Running the Program

There are two files that are used to solve universe environments: `universe_config` and `universe_solver.py`. 

You can run the simulation with the same parameters as `gym_solver`

`--max-steps`: The max number of steps to take per genome (timeout)

`--episodes`: The number of times to run a single genome. This takes the average fitness score over all episodes for one genome

`--render`: Renders the game while the algorithm is learning

`--generations`: The number of generations to evolve the network

`--checkpoint`: Uses a checkpoint to start the simulation

`--num-cores`: The number cores on your computer for parallel execution (not in `--render` mode)

`universe_solver.py` is initially setup to run with `flashgames.DriftRunners-v0`. You can run it with the following arguments:

```shell
python universe_solver.py --max-steps=10000 --generations=50 --render
```
### Simulating Other Games

You can also modify the program to run with any universe [environment](https://universe.openai.com/envs#flash_games).

Open up the `universe_solver.py` file. There is a section at the top with parameters that should be modified.

```
### User Params ###

# The name of the game to solve
game_name = 'flashgames.DriftRunners-v0'

# Change these to define the available actions in the game
action_sheet = [('KeyEvent', 'ArrowUp'), ('KeyEvent', 'ArrowLeft'), ('KeyEvent', 'ArrowRight')]

# Rules for actions that can't be taken at the same time
rules = [['ArrowLeft', 'ArrowRight'], ['ArrowUp', 'ArrowDown']]

### End User Params ###
```

`game_name` should be changed to the name of the game you want to simulate.

`action_sheet` holds all of the actions that can be taken in the game. A list of actions can be found [here](https://github.com/openai/universe/blob/master/universe/vncdriver/constants.py)

`rules` is an array of rules which are basically actions that can't be taken at the same time during one step. For example, the left and right key in a racing game cannot be pressed at the same time. 

Adjust these paramters according to the game enviroment you choose.

Next, open the `universe_config` and edit
```
output_nodes         = 3
```
to be the length of the action sheet array.

Also edit 
```
max_fitness_threshold   = 6000
```
to an appropriate target fitness score.

*Note* - `universe_solver.py` creates a checkpoint file like `gym_solver.py`
