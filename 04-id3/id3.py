from typing import NamedTuple
from collections import Counter
from math import log2 as log
from itertools import groupby


Sample = list[str]


class TrainingPair(NamedTuple):
    class_: str
    sample: Sample


def aggregate_by(iterable, key):
    """Groups elements from an iterable by key(elem).
    Analogous to itertools.group_by; however this function doesn't care about the order
    of the keys.
    """
    d = {}
    for elem in iterable:
        d.setdefault(key(elem), []).append(elem)
    return d



def entropy(pairs: list[TrainingPair]) -> float:
    classes_counter = Counter(pair.class_ for pair in pairs)
    proportions_of_classes = (i / len(pairs) for i in classes_counter.values())
    return -sum(ep * log(ep) for ep in proportions_of_classes)


def information_gain(attr: int, pairs, all_pairs_entropy: float) -> float:
    """Calculates the information gain after splitting the dataset
    on a given attribute.

    The information gain is the entropy of the whole set minus
    a weighted sum of entropies of all the subsets created by the split.
    """
    pairs_after_split = dict(groupby(pairs, lambda pair: pair.sample[attr]))
    print(pairs_after_split)


    return all_pairs_entropy - sum(
        entropy(subset) * len(subset) / len(pairs)
        for subset in pairs_after_split.values()
    )


def run_id3(pairs: list[TrainingPair]):
    all_pairs_entropy = entropy(pairs)
    best_attribute = max(
        range(len(pairs[0].sample)),
        key=lambda x: information_gain(x, pairs, all_pairs_entropy)
    )
    print(best_attribute)


if __name__ == "__main__":
    dataset = [
        TrainingPair("No", ["Sunny", "Hot", "High", "Weak"]),
        TrainingPair("No", ["Sunny", "Hot", "High", "Strong"]),
        TrainingPair("Yes", ["Overcast", "Hot", "High", "Weak"]),
        TrainingPair("Yes", ["Rain", "Mild", "High", "Weak"]),
        TrainingPair("Yes", ["Rain", "Cool", "Normal", "Weak"]),
        TrainingPair("No", ["Rain", "Cool", "Normal", "Strong"]), # 6
        TrainingPair("Yes", ["Overcast", "Cool", "Normal", "Strong"]), # 7
        TrainingPair("No", ["Sunny", "Mild", "High", "Weak"]), # 8
        TrainingPair("Yes", ["Sunny", "Cool", "Normal", "Weak"]), # 9
        TrainingPair("Yes", ["Rain", "Mild", "Normal", "Weak"]), # 10
        TrainingPair("Yes", ["Sunny", "Mild", "Normal", "Strong"]), # 11
        TrainingPair("Yes", ["Overcast", "Mild", "High", "Strong"]), # 12
        TrainingPair("Yes", ["Overcast", "Hot", "Normal", "Weak"]), # 13
        TrainingPair("No", ["Rain", "Mild", "High", "Strong"]), # 14
    ]
    total_entropy = entropy(dataset)
    print(total_entropy)
    print(run_id3(dataset))

