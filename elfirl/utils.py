import numpy as np

import logging
logger = logging.getLogger(__name__)

""" RL related utilities """


class Path():
    def __init__(self, transitions):
        self.transitions = transitions

    def append(self, transition):
        self.transitions.append(transition)

    def get_start_state(self):
        if len(self) < 1:
            raise ValueError("Path contains no transitions and thus no start state")
        return self.transitions[0].prev_state

    def __eq__(a, b):
        if len(a) != len(b):
            return False
        for t1, t2 in zip(a.transitions, b.transitions):
            if t1 != t2:
                return False
        return True

    def __len__(self):
        return len(self.transitions)

    def __repr__(self):
        ret = list()
        for t in self.transitions:
            ret.append("{};".format(t))
        return "".join(ret)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Path([transition.copy() for transition in self.transitions])


class Transition():
    def __init__(self, prev_state, action, next_state):
        self.prev_state = prev_state
        self.action = action
        self.next_state = next_state

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.prev_state, self.action, self.next_state).__hash__()

    def __repr__(self):
        return "T({}+{}->{})".format(self.prev_state, self.action, self.next_state)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Transition(self.prev_state.copy(), self.action, self.next_state.copy())


