from random import random, sample, randint, choices


def generate_edges(points):
    edges = []
    for index, point1 in enumerate(points):
        for point2 in points[index+1:]:
            edges.append((point1, point2))
    return edges


def generate_percent_of_edges(points, percent: float):
    edges = generate_edges(points)
    number_of_edges_to_remove = int((1-percent) * len(edges))
    for _ in range(number_of_edges_to_remove):
        edges.pop(randint(0, len(edges)-1))
    return edges


def chromosomes_gen(n, k, pop_init):
    lst = []
    for i in range(pop_init):
        chromosome = [0 for _ in range(n)]
        samples = sample(range(0, n), k=k)
        for j in range(k):
            chromosome[samples[j]] = 1
        lst.append(chromosome)
    return lst


def cost(cmbn, edges):
    obstacles = 0
    for e in edges:
        (u,v) = e
        if ((cmbn[u]==0) & (cmbn[v]==0)):
            obstacles += 1
    return obstacles


def selection(population, n, edges):
    return min(choices(population, k=n), key = lambda x: cost(x, edges))


def reproduce(population, edges):
    return [selection(population, 2, edges) for _ in range(len(population))]


def mutate_chromosome(chromosome, mutation_chance):
    if random() <= mutation_chance:
        pos1 = randint(0, len(chromosome)-1)
        pos2 = randint(0, len(chromosome)-1)
        while chromosome[pos1] != 0:
            pos1 = randint(0, len(chromosome)-1)
        while chromosome[pos2] != 1:
            pos2 = randint(0, len(chromosome)-1)
        chromosome[pos1], chromosome[pos2] = 1, 0
    return chromosome


def mutate_population(population, mutation_chance):
    return [mutate_chromosome(chromosome, mutation_chance) for chromosome in population]


def environment(vertices_num, chromosome_size, mutation_chance, pop_size, max_iterate, edges):
    population = chromosomes_gen(vertices_num, chromosome_size, pop_size)
    for it in range(max_iterate):
        population = mutate_population(reproduce(population, edges), mutation_chance)
        population.sort(key=lambda chromosome: cost(chromosome, edges))
        cost_value = cost(population[0], edges)
        if (it % 10) == 9:
            print("k = {}, Iteration = {}, Cost = {}".format(chromosome_size, it+1, cost_value))
        if cost_value == 0:
            break
    return cost_value, population[0]


def mfind(vertices_num,mutat_chance,pop_init,max_iterate,edges,start,end):
    result_dict = {}
    l = start
    h = end
    ans = 0
    while(l<=h):
        m = int((l+h)/2.0)
        cost_value,result = environment(vertices_num,m,mutat_chance,pop_init,max_iterate,edges)
        if(cost_value==0):
            result_dict[m] = result
            h = m-1
        else:
            l = m + 1
    return result_dict


print(chromosomes_gen(25, 5, 50))
# edges = generate_percent_of_edges([num for num in range(25)], 0.7)
# print(mfind(25, 0.05, 50, 500, edges, int(25/2), 25))
