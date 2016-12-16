import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel

generationRuns = 10
max_steps = 1000


def simulate_species(net, env, episodes=1, steps=5000, render=False):
    fitnesses = []
    cum_reward = 0.0
    for runs in range(episodes):
        inputs = my_env.reset()

        for j in range(steps):
            outputs = net.serial_activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, _ = env.step(action)
            if render:
                env.render()
            if done:
                break
            cum_reward += reward

        fitnesses.append(cum_reward)

    fitness = min(fitnesses)
    print("Species fitness: %s" % str(fitness))
    return fitness


def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, my_env, generationRuns, max_steps)


def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, generationRuns, max_steps)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    pop = population.Population(config_path)
    pe = parallel.ParallelEvaluator(8, worker_evaluate_genome)
    pop.run(pe.evaluate, 50)

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    import pickle
    with open('winner.pkl', 'wb') as output:
       pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    raw_input("Press Enter to continue...")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, env, 1, max_steps, render=True)

my_env = gym.make('CartPole-v0')
print "Input Nodes: " % str(len(my_env.observation_space.high))
print "Output Nodes: %s" % str(my_env.action_space.n)
train_network(my_env)
