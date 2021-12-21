import random


def elite_selection_model(generation):
    max_selected = int(len(generation) / 10)
    sorted_by_assess = sorted(generation, key=lambda x: x.fitness)
    return sorted_by_assess[:max_selected]


def tournament_selection(generation):
    parents = random.choices(generation, k=2)
    parents = sorted(parents, key=lambda x: x.fitness)
    return parents[0]