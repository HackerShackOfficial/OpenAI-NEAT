import argparse
import gym
import os
import numpy as np
import universe  # register the universe environments
from scipy import ndimage
from neat import nn, population, statistics, parallel


#action_sheet = [('KeyEvent', 'ArrowUp'), ('KeyEvent', 'ArrowDown'), ('KeyEvent', 'ArrowLeft'),
#                ('KeyEvent', 'ArrowRight'), ('KeyEvent', 'space')]

action_sheet = [('KeyEvent', 'ArrowUp'), ('KeyEvent', 'ArrowLeft'), ('KeyEvent', 'ArrowRight')]

parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=1000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--render', action='store_true')
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--num-cores', dest="numCores", type=int, default=4,
                    help="The number cores on your computer for parallel execution")
args = parser.parse_args()


def downsample_and_flatten(vision):
    new_obs = np.array(vision)
    # grayscale
    new_obs = new_obs.mean(axis=2)
    # downsample
    # new_obs = np.array(new_obs[::16, ::16])
    new_obs = np.array(block_mean(new_obs, 16))
    # 1d array
    new_obs = new_obs.flatten()
    return new_obs


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


def get_actions(outputs):
    actions = []
    for i in range(len(outputs)):
        if outputs[i] > 0:
            actions.append(action_sheet[i] + (True,))
        else:
            actions.append(action_sheet[i] + (False,))
    return actions


def check_rules():
    pass


def simulate_species(net, env, episodes=1, steps=5000, render=False):
    fitnesses = []
    for runs in range(episodes):
        inputs = my_env.reset()
        cum_reward = 0.0
        for j in range(steps):
            if inputs[0] is not None:
                new_obs = downsample_and_flatten(inputs[0]["vision"])
                outputs = net.serial_activate(new_obs)
            else:
                outputs = np.zeros(len(action_sheet)).tolist()

            inputs, reward, done, _ = env.step([get_actions(outputs) for ob in inputs])
            if render:
                env.render()
            if done[0]:
                break
            cum_reward += reward[0]

        fitnesses.append(cum_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness


def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, my_env, args.episodes, args.max_steps, render=args.render)


def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, args.episodes, args.max_steps, render=args.render)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'universe_config')
    pop = population.Population(config_path)
    # Load checkpoint
    if args.checkpoint:
        pop.load_checkpoint(args.checkpoint)
    # Start simulation
    #if args.render:
    pop.run(eval_fitness, args.generations)
    #else:
    #    pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)
    #    pop.run(pe.evaluate, args.generations)

    pop.save_checkpoint("checkpoint")

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    import pickle
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    raw_input("Press Enter to run the best genome...")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, env, 1, args.max_steps, render=True)

my_env = gym.make('flashgames.DriftRunners-v0')
my_env.configure(remotes=1)  # automatically creates a local docker container
observation_n = my_env.reset()
if args.render:
    my_env.render()

train_network(my_env)
