
import random
import math
from Graph import load_graph_from_file, path_cost_calculator


class SimulatedAnnealing:
    def __init__(self, graph, temperature, cooling_rate):
        self.graph = graph
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def run(self):
        current_path = self._initialize_path()
        best_path = current_path.copy()

        while self.temperature > 1:
            new_path = self._get_neighbor_path(current_path)
            current_cost = self._calculate_cost(current_path)
            new_cost = self._calculate_cost(new_path)

            if self._acceptance_probability(current_cost, new_cost) > random.random():
                current_path = new_path.copy()

            if self._calculate_cost(current_path) < self._calculate_cost(best_path):
                best_path = current_path.copy()

            self.temperature *= self.cooling_rate

        return best_path, path_cost_calculator(graph, best_path)

    def _initialize_path(self):
        nodes = list(self.graph.vertices.keys())
        start_node = random.choice(nodes)
        path = self._dfs_path(start_node)
        return path

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

                if len(stack) > len(self.graph.vertices):
                    # Add missing cities randomly to ensure all cities are covered
                    missing_cities = self._get_missing_cities(list(stack))
                    for city in missing_cities:
                        stack.append(city)
                        visited.add(city)

        return None


    def _get_neighbor_path(self, path):
        neighbor_path = path.copy()

        index1, index2 = random.sample(range(len(neighbor_path)), 2)
        index1, index2 = min(index1, index2), max(index1, index2)
        neighbor_path[index1:index2 + 1] = reversed(neighbor_path[index1:index2 + 1])

        missing_cities = self._get_missing_cities(neighbor_path)
        for city in missing_cities:
            self._insert_city(neighbor_path, city)

        neighbor_path = self._dfs_optimization(neighbor_path)

        return neighbor_path


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
            if vertex == end:
                return paths[end]
            for neighbor in graph.vertices[vertex].edges:
                if neighbor[0].name not in visited:
                    visited.append(neighbor[0].name)
                    stack.append(neighbor[0].name)
                    paths[neighbor[0].name] = paths[vertex] + [neighbor[0].name]

        return None


    def _calculate_cost(self, path):
      
        temp = path_cost_calculator(graph, path)
        if temp == 0:
            temp = 1e-15
        fitness = 1 / temp
           
        return fitness

    def _acceptance_probability(self, current_cost, new_cost):
        if new_cost > current_cost:
            return 1.0
        return math.exp((current_cost - new_cost) / self.temperature)

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
temperature = 1.00005
cooling_rate = 0.95
# start = time.time()
# sa = SimulatedAnnealing(graph, temperature, cooling_rate)
# best_path, best_cost = sa.run()
# end = time.time()
# print("Best Path:", best_path)
# print("Best Cost:", best_cost)
# print("Number of cities traveled:", len(best_path))
# print(end-start)
