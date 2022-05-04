from random import random, randint, choices, sample
from typing import TypeVar
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import itertools
import math


Edge = tuple[int, int]
Vertex = int


Population = list[type[Vertex]]
_Graph = TypeVar('_Graph')


def generate_edges(number_of_vertices: int) -> list[Edge]:
    """Generates all possible edges for a complete graph with a given size"""
    if number_of_vertices < 2:
        raise ValueError(
            "Cannot generate edges for graph with less than two vertices.")
    edges = []
    vertices = [number for number in range(number_of_vertices)]
    for index, point1 in enumerate(vertices):
        for point2 in vertices[index+1:]:
            edges.append((point1, point2))
    return edges


def generate_percent_of_edges(number_of_vertices: int, percent: float) -> list[Edge]:
    """Returns random graph with (1-percent)*100% removed edges.
    Every graph's vertex has at least one edge."""
    edges = generate_edges(number_of_vertices)
    number_of_edges_to_remove = int((1-percent) * len(edges))
    if len(edges) - number_of_edges_to_remove < math.ceil(number_of_vertices/2):
        raise ValueError(
            f"""Cannot remove {(1-percent)*100} percents of edges so that every vertex has at least one edge.
            Try increasing either graph's fullnes or number of graph's vertices."""
            )
    for _ in range(number_of_edges_to_remove):
        index = randint(0, len(edges)-1)
        occurrences = Counter(list(itertools.chain.from_iterable(edges)))
        while occurrences[edges[index][0]] == 1 or occurrences[edges[index][1]] == 1:
            index = randint(0, len(edges)-1)
        edges.pop(index)
    return edges


class Graph:
    def __init__(self, edges: list[Edge], vertices: list[Vertex]):
        self.vertices = vertices
        self.edges = edges
        self.cost = self.count_cost()

    def count_cost(self) -> int:
        """Calculates cost of the graph, i.e., number of uncovered egdes"""
        self.cost = len(list(
            filter(
                lambda edge: self.vertices[edge[0]] == 0 and self.vertices[edge[1]] == 0,
                self.edges
                )
            ))
        return self.cost

    @staticmethod
    def mutate(chromosome: type[_Graph], mutation_probability: float) -> type[_Graph]:
        """Swaps random covered vertex with random uncovered
           with a 'mutation_probability' chance"""
        if random() <= mutation_probability:
            length = len(chromosome.vertices)
            pos1 = randint(0, length-1)
            pos2 = randint(0, length-1)
            while chromosome.vertices[pos1] != 0:
                pos1 = randint(0, length-1)
            while chromosome.vertices[pos2] != 1:
                pos2 = randint(0, length-1)
            chromosome.vertices[pos1], chromosome.vertices[pos2] = 1, 0
        return chromosome

    @staticmethod
    def cover_random_vertex(vertices: list[Vertex]) -> list[Vertex]:
        index = randint(0, len(vertices)-1)
        while vertices[index] == 1:
            index = randint(0, len(vertices)-1)
        vertices[index] = 1
        return vertices

    @classmethod
    def make_from_edges(cls, edges: list[Edge],
                        number_of_vertices: int,
                        number_of_covered_vertices: int
                        ) -> type[_Graph]:
        """Returns Graph object with given number of covered vertices
           in random order."""
        chromosome = [0 for _ in range(number_of_vertices)]
        samples = sample(range(0, number_of_vertices), k=number_of_covered_vertices)
        for index in range(number_of_covered_vertices):
            chromosome[samples[index]] = 1
        return cls(edges, chromosome)

    def __repr__(self) -> str:
        return f"{self.vertices}"


class Evolution:
    def __init__(self, first_population: list[type[_Graph]],
                 mutation_probability: float, tournament_size: int,
                 iterations: int):
        self.population = first_population
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.iterations = iterations

    def tournament_selection(self, n: int = 2) -> type[_Graph]:
        """Picks n random points from the population and returns the one with the lowest cost."""
        return min(choices(self.population, k=n), key=lambda x: x.count_cost())

    def reproduce(self) -> list[type[_Graph]]:
        """Applies tournament selection to a population."""
        return [self.tournament_selection(self.tournament_size) for _ in range(len(self.population))]

    @staticmethod
    def mutate(population: list[type[_Graph]],
               mutation_probability: float) -> list[type[_Graph]]:
        """Calls mutate method on every chromosome of the population."""
        return [Graph.mutate(chromosome, mutation_probability) for chromosome in population]

    def run(self) -> list[type[_Graph]]:
        """Runs the evolution algorithm on a population
           and returns final population sorted by cost of every chromosome."""
        for it in range(self.iterations):
            self.population = self.mutate(self.reproduce(), self.mutation_probability)
            self.population.sort(key=lambda chromosome: chromosome.count_cost())
            cost_value = self.population[0].count_cost()
            # if (it % 10) == 9:
            #     print(f"k = {self.population[0].vertices.count(1)}, Iteration = {it+1}, Cost = {cost_value}")
            if cost_value == 0:
                break
        return self.population


def mfind(edges: list[Edge], number_of_vertices: int,
          size_of_population: int, mutation_probability: float,
          tournament_size: int, iterations: int, start: int) -> type[_Graph]:
    """Finds the minimum value of covered vertices to cover all the edges.
       Runs the evolution algorithm with the initial number of
       covered vertices equals to 'start'.
       After given number of iterations, if the solution was not found,
       increases number of covered vertices by 1.
       As a result returns the solution."""
    population = [Graph.make_from_edges(edges, number_of_vertices, start-1)
                  for _ in range(size_of_population)]
    while(start <= number_of_vertices):
        for i, chromosome in enumerate(population):
            chromosome.vertices = Graph.cover_random_vertex(chromosome.vertices)
        population_ev = Evolution(population, mutation_probability, tournament_size, iterations).run()
        if(population_ev[0].count_cost() == 0):
            break
        else:
            start += 1
    return population_ev[0]


def show_result(vertices: list[Vertex], edges: list[Edge],
                color_covered='red', color_uncovered='white', name: str = None):
    """Draws the graph."""
    G = nx.Graph()
    color_map = []
    for vertex, is_covered in enumerate(vertices):
        if is_covered == 1:
            color_map.append(color_covered)
        else:
            color_map.append(color_uncovered)
        G.add_node(vertex)
    for point1, point2 in edges:
        G.add_edge(point1, point2, weight=2)
    nx.draw_circular(G, node_color=color_map, with_labels=True)
    if name:
        plt.savefig(name)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        """Solve Vertex Cover Problem
        using genetic algorith."""
        )
    parser.add_argument('vertices', type=int, nargs=1,
                        help='total number of vertices in graph')
    parser.add_argument('covered', type=int, nargs=1,
                        help='total number of covered vertices in graph')
    parser.add_argument('population', type=int, nargs=1,
                        help='size of population')
    parser.add_argument('--fullness', action='store', default=0.5,
                        type=float,
                        help='percent of graph fullness (default: 0.5)')
    parser.add_argument('--mutation', metavar="PROBABILITY",
                        action='store', default=0.05, type=float,
                        help='probability that chromosomes will be mutated (default : 0.05)')
    parser.add_argument('--iterations', action='store', default=500,
                        type=int,
                        help="maximum number of iterarations (default: 500)")
    parser.add_argument('--tournament', metavar='SIZE', action='store', default=2,
                        type=int,
                        help="tournament size (default: 2)")
    parser.add_argument('--graph', action='store_true',
                        help='show the graph')
    args = parser.parse_args()

    NUMBER_OF_VERTICES = args.vertices[0]
    NUMBER_OF_COVERED_VERTICES = args.covered[0]
    SIZE_OF_POPULATION = args.population[0]
    PERCENT_OF_GRAPH_FULLNESS = args.fullness
    MUTATION_PROBABILITY = args.mutation
    MAX_ITERATIONS = args.iterations
    TOURNAMENT_SIZE = args.tournament

    COLOR_COVERED = 'red'
    COLOR_UNCOVERED = 'white'

    EDGES = generate_percent_of_edges(NUMBER_OF_VERTICES, PERCENT_OF_GRAPH_FULLNESS)
    vertices = mfind(EDGES, NUMBER_OF_VERTICES, SIZE_OF_POPULATION,
                     MUTATION_PROBABILITY, TOURNAMENT_SIZE, MAX_ITERATIONS,
                     NUMBER_OF_COVERED_VERTICES)
    print(vertices)
    if args.graph:
        show_result(vertices.vertices, EDGES, COLOR_COVERED, COLOR_UNCOVERED)