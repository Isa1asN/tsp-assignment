import sys
import time
from Graph import load_graph_from_file
import Hill
import Ga
import Sa


population_size = 500
num_generations = 2300
num_parents = 50
offspring_size = 50
mutation_rate = 0.01
temperature = 1.00005
cooling_rate = 0.95

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("The argument should be in this form: python main.py --algorithm ga --file my-file.txt")
        exit()

    algorithm = sys.argv[2]
    file = sys.argv[4]
    print(algorithm, file)

    graph = load_graph_from_file(file)

    if algorithm == 'ga':
        
        start = time.time()
        ga = Ga.GeneticAlgorithm(graph, population_size, num_generations, num_parents, offspring_size, mutation_rate)
        best_path, best_cost = ga.run()
        end = time.time()
        print("Best Path:", best_path)
        print("Best Cost:", best_cost)
        print("Number of cities traveled:", len(best_path))
        print(end-start)

    elif algorithm == 'sa':

        sa = Sa.SimulatedAnnealing(graph, temperature, cooling_rate)
        start = time.time()
        best_path, best_cost = sa.run()
        end = time.time()
        print("Best Path:", best_path)
        print("Best Cost:", best_cost)
        print("Number of cities traveled:", len(best_path))
        print(end-start)

    elif algorithm == 'hc':
        start = time.time()
        hc = Hill.HillClimbing(graph)
        best_path, best_cost = hc.run()
        end = time.time()
        print("Best Path:", best_path)
        print("Best Cost:", best_cost)
        print("Number of cities traveled:", len(best_path))
        print(end-start)
