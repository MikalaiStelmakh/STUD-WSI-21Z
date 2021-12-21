# import matplotlib.pyplot as plt
# from random import randint, random
# import networkx as nx
from collections import Counter
import itertools



# points = [(randint(0, 40), randint(0, 40)) for _ in range(25)]
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.clear()
# ax1.scatter(*zip(*points))

# fig.canvas.draw()
# fig.canvas.flush_events()
# plt.show(block=True)
# G = nx.Graph()
# NUMBER_OF_POINTS = 5
# CHANCE_OF_EDGE = 0.3
# EDGES = []


# for num1 in range(NUMBER_OF_POINTS-1):
#     if random() <= CHANCE_OF_EDGE:
#         EDGES.append((num1, num1+1))
#     for num in range(num1+2, NUMBER_OF_POINTS):
#         if random() <= CHANCE_OF_EDGE:
#             EDGES.append((num1, num))

# for point1, point2 in EDGES:
#     G.add_edge(f'{point1}', f'{point2}', weight=2)
# point = 1
# fitness = 0
# for edge in EDGES:
#     if point in edge:
#         fitness += 1

# print(EDGES)
# print(f'{fitness=}')
# nx.draw_circular(G)
# plt.show()


edges = [(1, 2), (2, 3), (3, 4)]
mylist2 = list(itertools.chain.from_iterable(edges))
print(Counter(mylist2)[1])
# vertex = 4
# print(list(filter(lambda x: vertex in x, edges)))

# for vertex, is_covered in vertices:

