from collections import Counter
from functools import partial
from math import log
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple,
                    Optional, Protocol, Sequence, TypeVar)
import csv

_T = TypeVar("_T")
_K = TypeVar("_K")

Sample = List[str]  # Sample is just a list of attributes


class Classifier(Protocol):
    """Describes the interface of a thing that can classify a given sample.
    The to_dot interface is used for debugging only, to view
    the 'guts' of a given classifier.
    """
    def classify(self, sample: Sample) -> str: ...
    def to_dot(self) -> str: ...


class DecisionTreeNode(NamedTuple):
    """DecisionTreeNode is a Classifier that recurses
    into other Classifiers based on the value of a single attribute.
    """
    attribute: int
    children: Dict[str, Classifier]

    def classify(self, sample: Sample) -> str:
        attr = sample[self.attribute]

        # Handle a special that "cant' happen" - evaluate by all nodes
        # and pick the most common class
        if attr not in self.children:
            class_counted = Counter(
                child.classify(sample)
                for child in self.children.values()
            )
            return class_counted.most_common(1)[0][0]

        else:
            return self.children[attr].classify(sample)

    def to_dot(self) -> str:
        self_name = f"node_{id(self)}"

        # Create a node for the current node
        x = f'  {self_name} [label="Split on attr {self.attribute}"];\n'

        # Create all the edges
        for edge_label, child in self.children.items():
            # Add dot from the child (to get its definition before inserting the edge)
            x += child.to_dot()

            # Add the edge to the child
            child_name = f"node_{id(child)}"
            escaped_edge_label = edge_label.encode("unicode_escape") \
                                           .decode("ascii") \
                                           .replace('"', r'\"')
            x += f'  {self_name} -> {child_name} [label="{escaped_edge_label}"];\n'

        return x


class DecisionTreeLeaf(NamedTuple):
    """DecisionTreeLeaf is a simple Classifier that always assignes a specific class"""
    class_: str

    def classify(self, sample: Sample) -> str:
        return self.class_

    def to_dot(self) -> str:
        class_escaped = self.class_.encode("unicode_escape") \
                                   .decode("ascii") \
                                   .replace('"', r'\"')
        return f'  node_{id(self)} [shape=box, label="{class_escaped}"];\n'


class TrainingPair(NamedTuple):
    """TrainingPair is just a sample (list of attributes)
    that has a known classification, and therefore is used for training.
    """
    class_: str
    sample: Sample


def aggregate_by(iterable: Iterable[_T], key: Callable[[_T], _K]) -> Dict[_K, List[_T]]:
    """Groups elements from an iterable by key(elem).
    Analogous to itertools.group_by; however this function doesn't care about the order
    of the keys.
    """
    d: Dict[_K, List[_T]] = {}
    for elem in iterable:
        d.setdefault(key(elem), []).append(elem)
    return d


def print_tree_as_dot(tree: Classifier) -> None:
    """Helper function that wraps tree.to_dot() in a digraph block,
    and prints the result (a fully valid dot file) onto stdout"""
    print("digraph {")
    print(tree.to_dot())
    print("}")


def entropy(pairs: Sequence[TrainingPair]) -> float:
    """Calculates the entropy of a dataset.

    The argument can be either a Counter instance for the classifications
    in a particular set; or the 'raw' set itself (in this case Counter object
    is created automatically).
    """
    classes_counter = Counter(pair.class_ for pair in pairs)
    proportions_of_classes = (i / len(pairs) for i in classes_counter.values())
    return -sum(ep * log(ep) for ep in proportions_of_classes)


def information_gain(attr: int, pairs: List[TrainingPair], all_pairs_entropy: float) -> float:
    """Calculates the information gain after splitting the dataset
    on a given attribute.

    The information gain is the entropy of the whole set minus
    a weighted sum of entropies of all the subsets created by the split.
    """
    pairs_after_split = aggregate_by(pairs, lambda pair: pair.sample[attr])
    return all_pairs_entropy - sum(
        entropy(subset) * len(subset) / len(pairs)
        for subset in pairs_after_split.values()
    )


def run_id3(pairs: List[TrainingPair], available_attrs: Optional[FrozenSet[int]] = None) \
        -> Classifier:
    """Creates a decision tree Classifier using the ID3 algorithm from a given
    training dataset.

    The available_attrs should be a frozenset of attributes to consider when
    creating a split. If None (or not provided), the algorithm will use all of the columns.

    It's mostly used in recursive calls, and should be probably provided as
    `frozenset( range(len(some_pair.sample)) )`; unless one of the attributes
    is the class itself, or some sort of identifier of a specific sample.
    """
    if available_attrs is None:
        available_attrs = frozenset(range(len(pairs[0].sample)))

    # Count classes
    class_counter = Counter(pair.class_ for pair in pairs)

    # All training pairs belong to a single class - create a leaf with that class
    if len(class_counter) == 1:
        return DecisionTreeLeaf(pairs[0].class_)

    # No attributes available - create a leaf with the most common class
    if not available_attrs:
        return DecisionTreeLeaf(class_counter.most_common(1)[0][0])

    best_attribute = max(
        available_attrs,
        key=partial(information_gain, pairs=pairs, all_pairs_entropy=entropy(pairs))
    )
    pairs_split_by_attr = aggregate_by(pairs, lambda pair: pair.sample[best_attribute])
    new_available_attrs = available_attrs.difference((best_attribute, ))

    return DecisionTreeNode(
        best_attribute,
        {attr_value: run_id3(new_pairs, new_available_attrs)
         for attr_value, new_pairs in pairs_split_by_attr.items()}
    )




def read_from_csv(path) -> list[TrainingPair]:
    with open(path, "r") as f:
        return [TrainingPair(sample[-1], sample[:-1])
                for sample in csv.reader(f)]


def evaluate_id3_on_dataset(samples, training_proportion: float):

        # Separate the samples
        cutoff = int(len(samples) * training_proportion)
        training_samples = samples[:cutoff]
        evaluate_samples = samples[cutoff+1:]

        # Run the ID3 algorithm
        available_attrs = frozenset(range(len(samples[0].sample)))
        decision_tree = run_id3(training_samples, available_attrs)

        # Evaluate the algorithm
        all_classes = {sample.class_ for sample in samples}
        correct_classifications = 0

        # Results are (initially) a mapping got_class -> expected_class -> count;
        # Additional "Expected \ Got" expected_class is a special key to pretty-print the table
        # After processing the count is replaced by a percentage
        results = {
            expected_class: {"Spodziewana \\ Otrzymana": expected_class} |
            {got_class: 0 for got_class in all_classes}
            for expected_class in all_classes
        }

        for sample in evaluate_samples:
            generated_class = decision_tree.classify(sample.sample)

            assert isinstance(results[sample.class_][generated_class], int)
            results[sample.class_][generated_class] += 1  # type: ignore

            if decision_tree.classify(sample.sample) == sample.class_:
                correct_classifications += 1

        # Convert results to percentages (as string for pretty printing)
        # for got_classes in results.values():
        #     for got_class, count in got_classes.items():
        #         if not isinstance(count, str):
        #             got_classes[got_class] = f"{count / len(evaluate_samples):.4%}"

        return correct_classifications / len(evaluate_samples), len(samples)


if __name__ == "__main__":
    dataset = read_from_csv("lab4/car.data")
    # print(run_id3(dataset))
    print(evaluate_id3_on_dataset(dataset, 0.6))
    # run_id3(
    # [
    # TrainingPair("No", ["Sunny", "Hot", "High", "Weak"]),
    # TrainingPair("No", ["Sunny", "Hot", "High", "Strong"]),
    # TrainingPair("Yes", ["Overcast", "Hot", "High", "Weak"]),
    # TrainingPair("Yes", ["Rain", "Mild", "High", "Weak"]),
    # TrainingPair("Yes", ["Rain", "Cool", "Normal", "Weak"]),
    # TrainingPair("No", ["Rain", "Cool", "Normal", "Strong"]), # 6
    # TrainingPair("Yes", ["Overcast", "Cool", "Normal", "Strong"]), # 7
    # TrainingPair("No", ["Sunny", "Mild", "High", "Weak"]), # 8
    # TrainingPair("Yes", ["Sunny", "Cool", "Normal", "Weak"]), # 9
    # TrainingPair("Yes", ["Rain", "Mild", "Normal", "Weak"]), # 10
    # TrainingPair("Yes", ["Sunny", "Mild", "Normal", "Strong"]), # 11
    # TrainingPair("Yes", ["Overcast", "Mild", "High", "Strong"]), # 12
    # TrainingPair("Yes", ["Overcast", "Hot", "Normal", "Weak"]), # 13
    # TrainingPair("No", ["Rain", "Mild", "High", "Strong"]), # 14
    # ]
    # )

