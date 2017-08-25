import numpy as np
import scipy as sp

from elfirl.utils import Path, Transition

from pybrain.structure.modules.module import Module
from pybrain.rl.environments import Environment, EpisodicTask
from pybrain.rl.explorers.discrete import EpsilonGreedyExplorer
from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

import logging
logger = logging.getLogger(__name__)

""" Extensions to the pybrain library. """


class ParametricLoggingEpisodicTask(EpisodicTask):
    """ Extension of the basic episodic task with tunable parameters
        and better support for logging.
    """

    def __init__(self, env):
        super(ParametricLoggingEpisodicTask, self).__init__(env)
        self.env.task = self
        self.v = None

    def setup(self, variables):
        """ Set the variables of the task
        """
        self.v = variables

    def clean(self):
        pass


class ParametricLoggingEnvironment(Environment):
    """ Extension of the basic environment with tunable parameters
        and better support for logging.
    """

    def __init__(self):
        super(ParametricLoggingEnvironment, self).__init__()
        self.v = None
        self.log = None
        self.task = None  # set by task
        self.state = None
        self.prev_state = None
        self.action = None
        self.log_session_variables = list()  # logged at start of session
        self.log_step_variables = list()  # logged after each step

    def setup(self, variables, random_state):
        """ Finishes the initialization
        """
        self.v = variables
        self.random_state = random_state
        self.reset()
        print("ENV V", self.v)

    def start_logging(self):
        self.log = dict()

    def end_logging(self):
        self.log = None

    def _start_log_for_new_session(self):
        """ Set up log when new session starts
        """
        if self.log != None:
            if "session" not in self.log:
                self.log["session"] = 0
                self.log["sessions"] = [dict()]
            else:
                self.log["session"] += 1
                self.log["sessions"].append(dict())
            self.step_data = self.log["sessions"][self.log["session"]]
            for varname in self.log_session_variables:
                self.step_data[varname] = getattr(self, varname)
            for varname in self.log_step_variables:
                if varname == "path":
                    self.step_data["path"] = Path([])
                else:
                    self.step_data[varname] = list()

    def _log_transition(self):
        """ Should be called after transition
        """
        if self.log != None:
            for varname in self.log_step_variables:
                if varname == "rewards":
                    self.step_data["rewards"].append(self.task.getReward())
                elif varname == "path":
                    self.step_data["path"].append(Transition(self.prev_state, self.action, self.state))
                else:
                    self.step_data[varname].append(getattr(self, varname))

    def clean(self):
        pass


class SparseActionValueTable(ActionValueTable):
    """ Sparse version of the ActionValueTable from pybrain, uses less memory.
    Also uses the provided random_state for consistency.
    """

    def __init__(self, numActions, random_state, name=None):
        Module.__init__(self, 1, 1, name)
        self.n_actions = numActions
        self.numColumns = numActions
        self.random_state = random_state
        if isinstance(self, Module) or isinstance(self, Connection):
            self.hasDerivatives = True
        if self.hasDerivatives:
            self._derivs = None
        self.randomize()
        self._params = None

    def _forwardImplementation(self, inbuf, outbuf):
        """ Take a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(inbuf[0])

    def randomize(self):
        self.sparse_params = dict() # dictionary-of-rows sparse matrix
        self.initval = None

    def initialize(self, value=1e-5):
        self.initval = value

    def getMaxAction(self, state):
        values = self.getActionValues(state)
        action = sp.where(values == max(values))[0]
        return self.random_state.choice(action)

    def check_bounds(self, column=None):
        if column < 0 or column >= self.n_actions:
            raise ValueError("Column out of bounds (was {})".format(column))

    def getValue(self, row, column):
        return self.getActionValues(row)[column]

    def _init_or_random_val(self):
        if self.initval is None:
            # From ParameterContainer.randomize()
            return self.random_state.randn() * self.stdParams
        else:
            # From ActionValueTable.initialize()
            return self.initval

    def getActionValues(self, state):
        if state is None:
            return None
        r = self.sparse_params.get(state, None)
        if r is None:
            r = np.array([float(self._init_or_random_val()) for i in range(self.n_actions)])
            self.sparse_params[state] = r
        return r[:]

    def updateValue(self, row, column, value):
        self.check_bounds(column)
        if row is None or column is None:
            return
        r = self.getActionValues(row)
        r[column] = value
        self.sparse_params[row] = r

    def mutate(self):
        raise NotImplementedError("This should not be called.")

    def derivs(self):
        raise NotImplementedError("This should not be called.")

    def resetDerivatives(self):
        raise NotImplementedError("This should not be called.")



class EpisodeQ(Q):
    """ A slight modification of the pybrain Q learner to add special handling
        for the end state of the session.
    """

    def __init__(self, alpha=1.0, w=1.0, gamma=0.99, iters=10):
        ValueBasedLearner.__init__(self)

        self.alpha = alpha  # step scale
        self.w = w  # learning rate
        self.gamma = gamma  # temporal discount
        self.iters = 10  # number of times to propagate value changes

        self.step = 0

    def learn(self):
        alpha = self.alpha / ((self.step + 1) ** self.w)
        self.step += 1
        rewards = list()
        pos_end = list()
        path_len = list()

        for i in range(self.iters):
            for seq in self.dataset:
                nextstate = None
                nextaction = None
                nextreward = None
                if i == 0:
                    seq_reward = 0
                    n_steps = 0

                for state, action, reward in reversed(list(seq)):  # zip is not directly reversible

                    state = int(state)
                    action = int(action)
                    reward = float(reward)
                    qvalue = float(self.module.getValue(state, action))

                    if nextstate == None:
                        qvalue += alpha * (reward - qvalue)
                        if i == 0:
                            if reward > 0:
                                pos_end.append(1)
                            else:
                                pos_end.append(0)
                    else:
                        maxnext = float(max(self.module.getActionValues(nextstate)))
                        qvalue += alpha * (reward + self.gamma * maxnext - qvalue)

                    self.module.updateValue(state, action, qvalue)
                    nextstate = state
                    nextaction = action
                    nextreward = reward
                    if i == 0:
                        seq_reward += reward
                        n_steps += 1

                if i == 0:
                    path_len.append(n_steps)
                    rewards.append(seq_reward)

        print("step {} reward mean={:.1f} (std={:.1f}) | positive final reward in {:.2f}% | path len mean={:.2f} (std={:.2f})".format(
                self.step, np.mean(rewards), np.std(rewards), 100*sum(pos_end)/float(len(pos_end)), np.mean(path_len), np.std(path_len)))


class EGreedyExplorer(DiscreteExplorer):
    """ Reimplementation of rllib EpsilonGreedyExplorer to use the provided random_state
    """
    def __init__(self, random_state, epsilon = 0.3, decay = 0.9999):
        DiscreteExplorer.__init__(self)
        self.random_state = random_state
        self.epsilon = epsilon
        self.decay = decay
        self.module = None

    def _forwardImplementation(self, inbuf, outbuf):
        assert self.module is not None

        if self.random_state.rand() < self.epsilon:
            outbuf[:] = np.array([self.random_state.randint(self.module.numActions)])
        else:
            outbuf[:] = inbuf

        self.epsilon *= self.decay


class Policy():
    """ Encapsulated policy
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, state, action):
        """ Returns p(action | state) accoring to deterministic policy with ties broken arbitrarily
        """
        s = float(state.__hash__())  # pybrain secretly casts state to float when we do rl
        a = int(action)
        qvalues = self.module.getActionValues(s)
        maxq = max(qvalues)
        if qvalues[a] == maxq:
            n_max = sum([1 if q == maxq else 0 for q in qvalues])
            return 1.0 / n_max
        return 0

