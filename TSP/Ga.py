
import time
from Graph import load_graph_from_file, path_cost_calculator
import random

class GeneticAlgorithm:
    def __init__(self, graph, population_size, num_generations, num_parents, offspring_size, mutation_rate):
        self.graph = graph
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate

    def run(self):
        population = self._initialize_population()
        for generation in range(self.num_generations):
            fitness_scores = self._calculate_fitness(population)
            parents = self._selection(population, fitness_scores)
            offspring = self._crossover(parents)
            population = self._mutation(offspring)

            best_path = max(population, key=lambda path: path_cost_calculator(graph, path))
            for path in population:
                num_cities = len(path)
                if 28 <= num_cities <= 39:
                    best_path = path
            best_cost = path_cost_calculator(self.graph, best_path)

            print("Generation:", generation + 1, "Best Cost:", best_cost)
        return best_path, best_cost


    def _initialize_population(self):
        population = []
        nodes = list(self.graph.vertices.keys())

        for _ in range(self.population_size):
            # Generate a random path using DFS
            start_node = random.choice(nodes)
            path = self._dfs_path(start_node)

            population.append(path)

        return population


    def _dfs_path(self, start):
        stack = [start]
        visited = {start}

        while stack:
            node = stack[-1]
            neighbors = self.graph.vertices[node].edges
            unvisited_neighbors = [
                neighbor[0].name for neighbor in neighbors if neighbor[0].name not in visited
            ]

            if not unvisited_neighbors:
                stack.pop()
            else:
                next_node = random.choice(unvisited_neighbors)
                stack.append(next_node)
                visited.add(next_node)

                if len(visited) == len(self.graph.vertices):
                    # Once all nodes are visited, return the path
                    return list(stack)

        return None

    def _calculate_fitness(self, population):
        fitness_scores = []
        for path in population:
            temp = path_cost_calculator(graph, path)
            if temp == 0:
                temp = 1e-15
            fitness = 1 / temp
            fitness_scores.append(fitness)
        return fitness_scores

    def _selection(self, population, fitness_values):
        selected_parents = []
        population_size = len(population)
        for _ in range(population_size):
            tournament = random.choices(range(population_size), k=self.num_parents)
            tournament_fitness = [fitness_values[i] for i in tournament]
            best_index = tournament[tournament_fitness.index(min(tournament_fitness))]
            selected_parents.append(population[best_index])            
        return selected_parents

    def _crossover(self, parents):
        offspring = []
        for _ in range(self.offspring_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point1 = random.randint(1, len(parent1) - 1)
            crossover_point2 = random.randint(1, len(parent2) - 1)

            if crossover_point1 > crossover_point2:
                crossover_point1, crossover_point2 = crossover_point2, crossover_point1

            child = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]

            child = self._validate_path(child)
            missing_cities = self._get_missing_cities(child)
            for city in missing_cities:
                self._insert_city(child, city)

            offspring.append(child)

        return offspring

    def _mutation(self, population):
        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                mutated_path = population[i].copy()
                index1, index2 = random.sample(range(len(mutated_path)), 2)
                index1, index2 = min(index1, index2), max(index1, index2)
                mutated_path[index1:index2 + 1] = reversed(mutated_path[index1:index2 + 1])

                mutated_path = self._validate_path(mutated_path)
                missing_cities = self._get_missing_cities(mutated_path)
                for city in missing_cities:
                    self._insert_city(mutated_path, city)

                mutated_path = self._dfs_optimization(mutated_path)

                if len(mutated_path) > 20:
                    population[i] = mutated_path

        return population

    def _dfs_optimization(self, path):
        modified_path = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            dfs_path = self._dfs(self.graph, start, end)
            if dfs_path:
                modified_path += dfs_path[:-1]
            else:
                modified_path += [start, end]
        modified_path.append(path[-1])
        return modified_path

    def _dfs(self, graph, start, end):
        stack = [start]
        visited = [start]
        paths = {start: [start]}

        while stack:
            vertex = stack.pop()
            for neighbor in graph.vertices[vertex].edges:
                if neighbor[0].name not in visited:
                    visited.append(neighbor[0].name)
                    stack.append(neighbor[0].name)
                    paths[neighbor[0].name] = paths[vertex] + [neighbor[0].name]
                    if neighbor[0].name == end:
                        return paths[end]

        return None


    def _get_missing_cities(self, path):
        all_cities = set(self.graph.vertices.keys())
        visited_cities = set(path)
        missing_cities = all_cities - visited_cities
        return missing_cities

    def _insert_city(self, path, city):
        min_cost = float('inf')
        min_index = -1
        for i in range(1, len(path)):
            node1 = self.graph.vertices[path[i - 1]]
            node2 = self.graph.vertices[path[i]]
            edge_weight = 0
            for edge in node1.edges:
                if edge[0] == node2:
                    edge_weight = edge[1]
                    break
            if edge_weight < min_cost:
                min_cost = edge_weight
                min_index = i

        if min_index != -1:
            path.insert(min_index, city)

    def _validate_path(self, path):
        unique_nodes = set()
        valid_path = []
        for node in path:
            if node not in unique_nodes:
                valid_path.append(node)
                unique_nodes.add(node)
        return valid_path


graph = load_graph_from_file('cities.txt')
population_size = 500
num_generations = 2300
num_parents = 50
offspring_size = 50
mutation_rate = 0.01

# start = time.time()
# ga = GeneticAlgorithm(graph, population_size, num_generations, num_parents, offspring_size, mutation_rate)
# best_path, best_cost = ga.run()
# end = time.time()

# print("Best Path:", best_path)
# print("Best Cost:", best_cost)
# print("Number of cities traveled:", len(best_path))
# print(end-start)
