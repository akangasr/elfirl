import numpy as np
import gc

from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from elfirl.pybrain_extensions import SparseActionValueTable, EpisodeQ, EGreedyExplorer, Policy

import logging
logger = logging.getLogger(__name__)

""" Generic RL model """


class RLParams():
    def __init__(self,
            n_training_episodes=1,
            n_episodes_per_epoch=1,
            n_simulation_episodes=1,
            q_alpha=1.0,
            q_w=1.0,
            q_gamma=0.98,
            exp_epsilon=0.1,
            exp_decay=1.0):
        self.n_training_episodes = n_training_episodes
        self.n_episodes_per_epoch = n_episodes_per_epoch
        self.n_simulation_episodes = n_simulation_episodes
        self.q_alpha = q_alpha
        self.q_w = q_w
        self.q_gamma = q_gamma
        self.exp_epsilon = exp_epsilon
        self.exp_decay = exp_decay

    def to_dict(self):
        self.__dict__


class RLModel():

    def __init__(self,
            rl_params,
            parameter_names,
            env,
            task,
            clean_after_call=False):
        """

        Parameters
        ----------
        rl_params : RLParams
        parameters : parameter names in order
        env : Environment model
        task : EpisodecTask instance
        clean_after_call: bool
        """
        self.rl_params = rl_params
        self.parameter_names = parameter_names
        self.env = env
        self.task = task
        self.agent = None
        self.clean_after_call = clean_after_call

    def to_dict(self):
        return {
                "rl_params": self.rl_params.to_dict(),
                "parameters": self.parameters,
                }

    def train_model(self, parameter_values, random_state=None):
        self._parameters(parameter_values)
        self._build_model(random_state)
        self._train_model()

    def __call__(self, *parameter_values, index_in_batch=None, random_state=None):
        """ Simulates data.
        Interfaces to ELFI as a sequential simulator.

        Parameters
        ----------
        parameter_values : list of model variables
            Length should equal length of parameters
        random_state: random number generator

        Returns
        -------
        Simulated trajectories as a dict
        """
        self.train_model(parameter_values, random_state=random_state)
        log_dict = self.simulate(random_state)
        if self.clean_after_call is True:
            self.clean()
        return log_dict

    def get_policy(self):
        """ Returns the current policy of the agent
        """
        return self.agent.get_policy()

    def _parameters(self, parameter_values):
        """ Parse parameter values
        """
        self.p = dict()
        if len(self.parameter_names) != len(parameter_values):
            raise ValueError("Number of model variables was {} ({}), expected {}"
                    .format(len(parameter_values), parameter_values, len(self.parameter_names)))
        for name, val in zip(self.parameter_names, parameter_values):
            self.p[name] = val
        logger.debug("Model parameters: {}".format(self.p))

    def _build_model(self, random_state):
        """ Initialize the model
        """
        self.env.setup(self.p, random_state)
        self.task.setup(self.p)
        outdim = self.task.env.outdim
        n_actions = self.task.env.numActions
        self.agent = RLAgent(outdim,
                n_actions,
                random_state,
                rl_params=self.rl_params)
        logger.debug("Model initialized")

    def _train_model(self):
        """ Uses reinforcement learning to find the optimal strategy
        """
        self.experiment = EpisodicExperiment(self.task, self.agent)
        n_epochs = int(self.rl_params.n_training_episodes / self.rl_params.n_episodes_per_epoch)
        logger.debug("Fitting user model over {} epochs, each {} episodes, total {} episodes."
                .format(n_epochs, self.rl_params.n_episodes_per_epoch, n_epochs*self.rl_params.n_episodes_per_epoch))
        for i in range(n_epochs):
            logger.debug("RL epoch {}".format(i))
            self.experiment.doEpisodes(self.rl_params.n_episodes_per_epoch)
            self.agent.learn()
            self.agent.reset()  # reset buffers

    def simulate(self, random_state):
        """ Simulates agent behavior in 'n_sim' episodes.
        """
        logger.debug("Simulating user actions ({} episodes)".format(self.rl_params.n_simulation_episodes))
        self.experiment = EpisodicExperiment(self.task, self.agent)

        # set training flag off
        self.task.env.training = False
        # deactivate learning for experiment
        self.agent.learning = False
        # deactivate exploration
        explorer = self.agent.learner.explorer
        self.agent.learner.explorer = EGreedyExplorer(epsilon=0, decay=1, random_state=random_state)
        self.agent.learner.explorer.module = self.agent.module
        # activate logging
        self.task.env.start_logging()

        # simulate behavior
        self.experiment.doEpisodes(self.rl_params.n_simulation_episodes)
        # store log data
        dataset = self.task.env.log

        # deactivate logging
        self.task.env.end_logging()
        # reactivate exploration
        self.agent.learner.explorer = explorer
        # reactivate learning for experiment
        self.agent.learning = True
        # set training flag back on
        self.task.env.training = True

        return dataset

    def clean(self):
        self.agent = None
        self.env.clean()
        self.task.clean()
        gc.collect()


class RLAgent(LearningAgent):
    def __init__(self, outdim, n_actions, random_state, rl_params):
        """ RL agent
        """
        module = SparseActionValueTable(n_actions, random_state)
        module.initialize(0.0)
        learner = EpisodeQ(alpha=rl_params.q_alpha,
                           w=rl_params.q_w,
                           gamma=rl_params.q_gamma)
        learner.explorer = EGreedyExplorer(random_state,
                                           epsilon=rl_params.exp_epsilon,
                                           decay=rl_params.exp_decay)
        LearningAgent.__init__(self, module, learner)

    def get_policy(self):
        return Policy(self.module)

