import enum
import random
from typing import Callable, NamedTuple, Tuple
from collections import deque


class Field(enum.IntEnum):
    """Field is a type used to represent field types of a single map."""
    START = ord("S")
    FREE = ord(" ")
    HOLE = ord("H")
    GOAL = ord("G")


class Action(enum.IntEnum):
    """Action is an enum used to represent available actions."""
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    def on_field(self, row: int, col: int, side_len: int) -> Tuple[int, int]:
        """Performs an action and returns new agent's position."""
        if self == Action.LEFT:
            col = max(col - 1, 0)
        elif self == Action.RIGHT:
            col = min(col + 1, side_len - 1)
        elif self == Action.UP:
            row = max(row - 1, 0)
        elif self == Action.DOWN:
            row = min(row + 1, side_len - 1)
        return row, col

    def sample() -> int:
        """Returns random action."""
        return random.sample([Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP], k=1)[0]


class ActionResult(NamedTuple):
    """Represents a result of an agent performing an action."""
    state: int
    reward: float
    done: bool


def reward_default(field: Field) -> float:
    """The default reward function for the QUber problem -
    returns 1.0 if the goal is reached; 0 otherwise."""
    return float(field == Field.GOAL)


def reward_alternative_1(field: Field) -> float:
    """Custom reward function that punishes falling into holes."""
    if field == field.GOAL:
        return 1.0
    elif field == field.HOLE:
        return -1.0
    else:
        return 0.0


def reward_alternative_2(field: Field) -> float:
    """Custom reward reward function that punishes falling into halls,
    but gives a much higher reward for reaching the goal."""
    if field == field.GOAL:
        return 10.0
    elif field == field.HOLE:
        return -1.0
    else:
        return 0.0


class Environment:
    def __init__(self, side_length: int = 8, pos_start: int = None,
                 pos_goal: int = None, probability_of_hole: float = 0.25,
                 reward: Callable[[Field], float] = reward_alternative_1) -> None:
        self.pos_start = pos_start if pos_start else 0
        self.pos_goal = pos_goal if pos_goal else (side_length**2)-side_length
        self.side_length = side_length
        self.total_fields = side_length ** 2
        self.map = self._generate_map(self.pos_start, self.pos_goal, probability_of_hole)
        while(not self._check_map()):
            self.map = self._generate_map(self.pos_start, self.pos_goal, probability_of_hole)

        self.state: int = self.pos_start
        self.reward = reward

        self.outcomes = {}

        # Fill the table of next steps
        for row in range(self.side_length):
            for col in range(self.side_length):
                field_index: int = row * self.side_length + col
                field = Field(self.map[field_index])
                self.outcomes[field_index] = {}

                for direction in Action:

                    if field == Field.GOAL or field == Field.HOLE:
                        # Don't allow to move out from an end field
                        result = ActionResult(field_index, 0.0, True)

                    else:
                        result = self._result_of_move(row, col, direction)

                    self.outcomes[field_index][direction] = result

    def _generate_map(self, pos_start: int, pos_goal: int,
                      probability_of_hole: float) -> list[Field]:
        """Generate a map. Each field is a hole with given probability."""
        map = []
        for i in range(self.total_fields):
            if i == pos_start:
                map.append(Field.START)
            elif i == pos_goal:
                map.append(Field.GOAL)
            elif random.random() < probability_of_hole:
                map.append(Field.HOLE)
            else:
                map.append(Field.FREE)
        return map

    def _result_of_move(self, row: int, col: int, action: Action) -> ActionResult:
        """Creates an ActionResult object for an action performed by an action
        in a specific state."""
        new_row, new_col = action.on_field(row, col, self.side_length)
        new_field_idx = new_row * self.side_length + new_col
        new_field = Field(self.map[new_field_idx])
        reward = self.reward(new_field)
        return ActionResult(
            new_field_idx,
            reward,
            new_field == Field.GOAL or new_field == Field.HOLE,
        )

    def _check_map(self) -> None:
        """Asserts several assumptions about the map."""
        return len(self.map) == self.total_fields and \
            sum(1 for f in self.map if f == Field.GOAL) > 0 and \
            sum(1 for f in self.map if f == Field.START) == 1 and \
            Graph.make_from_env(self).isReachable(self.pos_start, self.pos_goal)

    def reset(self) -> None:
        """Resets the actor's state, without modifying the environment."""
        self.state = self.pos_start

    def step(self, action: Action) -> ActionResult:
        """Performs an action. Returns the result and updates the actor's state."""
        result = self.outcomes[self.state][action]
        self.state = result.state
        return result


class GameInterface:
    BORDER = '\u001b[35m'
    HOLE = '\u001b[31m'
    GOAL = '\u001b[32m'
    START = '\u001b[33m'
    RESET = '\u001b[0m'
    BOLD = '\033[1m'
    RESET_FONT = '\033[0m'

    def __init__(self, env: Environment, path: list[int] = None) -> None:
        self.env = env
        self.path = path if path else []

    def _print_map(self) -> str:
        """Prints map with borders."""
        map = self.env.map
        result = self.BORDER + "-"*(self.env.side_length+2) + "\n|"
        for cur_pos, field in enumerate(map):
            used = False
            if cur_pos % self.env.side_length == 0 and cur_pos != 0:
                result += self.BORDER + "|\n|"
            for pos in self.path:
                if pos == cur_pos:
                    char = self.RESET + '.'
                    used = True
            if field == Field.START:
                char = self.START + "S"
            elif field == Field.GOAL:
                if used:
                    char = self.GOAL + self.BOLD + "G" + self.RESET_FONT
                else:
                    char = self.GOAL + "G"
            elif field == Field.HOLE:
                if used:
                    char = self.HOLE + self.BOLD + "x" + self.RESET_FONT
                else:
                    char = self.HOLE + "x"
            if field == Field.FREE and not used:
                char = self.RESET + " "
            result += char
        result += self.BORDER + "|\n" + "-"*(self.env.side_length+2) + self.RESET
        print(result)
        return result


class Graph:
    """Class needed to check if the path between
    start and goal exists."""
    def __init__(self, n) -> None:
        self.adj = [[] for i in range(n)]

    def addEdge(self, v: int, w: int) -> None:
        self.adj[v].append(w)
        self.adj[w].append(v)

    def isReachable(self, s: int, d: int) -> bool:
        """Checks if path between two vertices exists."""
        if (s == d):
            return True

        visited = [False for i in self.adj]
        queue = deque()
        visited[s] = True
        queue.append(s)

        while (len(queue) > 0):
            s = queue.popleft()
            for i in set(self.adj[s]):

                if (i == d):
                    return True

                if (not visited[i]):
                    visited[i] = True
                    queue.append(i)
        return False

    @classmethod
    def make_from_env(cls, env: Environment):
        """Make graph from Environment class object."""
        graph = cls(env.total_fields)
        for i in range(env.total_fields):
            i1 = i + env.side_length
            if i1 < env.total_fields and env.map[i] != Field.HOLE \
               and env.map[i1] != Field.HOLE:
                graph.addEdge(i, i1)
            if (i+1) % env.side_length != 0 and env.map[i] != Field.HOLE \
               and env.map[i+1] != Field.HOLE:
                graph.addEdge(i, i+1)
        return graph


def actions_to_string(actions: list[Action]) -> str:
    act_to_str = {
        0: 'LEFT',
        1: 'DOWN',
        2: 'RIGHT',
        3: 'UP',
    }
    return ", ".join([act_to_str[action] for action in actions])


if __name__ == "__main__":
    env = Environment(probability_of_hole=0.1, reward=reward_alternative_1)
