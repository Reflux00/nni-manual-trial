import numpy as np
import gym
from .SAC_model import SAC_Agent
from .ReplayBuffer import RandomBuffer, device
from nni.tuner import Tuner
from nni.typehint import Parameters, SearchSpace, TrialRecord
from nni.utils import extract_scalar_reward, OptimizeMode
import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


def json_to_gym_space(json_space):
    
    low = []
    high = []
    for key, value in json_space.items():

        assert value['_type'] in ['uniform'], 'Only support Box space'
        space = value['_value']

        low.append(space[0])
        high.append(space[1])

    dim = len(low)

    return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32, shape=(dim,))

class SimpleENv(gym.Env):

    def __init__(self, search_space:dict):

        self.action_space = json_to_gym_space(search_space)
        self.observation_space = json_to_gym_space(search_space)

        self.optimization_dim = len(search_space)


    def step(self, action:np.ndarray):
        assert action.shape == (self.optimization_dim, ), "action.shape should be equal to (self.optimization_dim, )"
        reward = 0
        done = False
        info = {}
        return np.zeros(self.optimization_dim), reward, done, info

    def reset(self):
        return np.zeros(self.optimization_dim)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class Action_adapter(object):

    low: np.ndarray = None
    high: np.ndarray = None
    space_size: np.ndarray = None
    
    def __init__(self, search_space:dict):
        
        self._search_space_to_action_space(search_space)

    def update_search_space(self, search_space:dict):
        self._search_space_to_action_space(search_space)

    def _search_space_to_action_space(self, search_space:dict):
        low = []
        high = []
        for key, value in search_space.items():

            assert value['_type'] in ['uniform'], 'Only support Box space'
            space = value['_value']

            low.append(space[0])
            high.append(space[1])

        if self.low is not None:
            assert len(low) == len(self.low), "search_space dim should be equal to action_space dim"

        low = np.array(low)
        high = np.array(high)
        assert np.all(low < high), "low should be less than high"

        self.low = low
        self.high = high
        self.space_size = high - low

    def __call__(self, action):
        """
        Scales the input action between low and high.

        Parameters:
        action (float): The action to be scaled, between -1 and 1.

        Returns:
        float: The scaled action between low and high in search space.
        """
        assert action.shape == (len(self.low), ), "action.shape should be equal to low and high"
        # Scale the action between low and high
        scaled_action = ((action + 1) / 2) * self.space_size + self.low

        return scaled_action

    
class Action_adapter_reverse(Action_adapter):
    
    def __call__(self, action):

        """
        Reverses the scaling of the input action between low and high.

        Parameters:
        scaled_action (float): The scaled action to be reversed.

        Returns:
        float: The reversed action between -1 and 1.
        """
           
        assert action.shape == (len(self.low), ), "action.shape should be equal to low and high"
        # Scale the action between low and high
        scaled_action = ((action - self.low) / self.space_size) * 2 - 1

        return scaled_action


class SACArguments(NamedTuple):
    gamma:float = 0.99
    hid_shape:tuple[int, int] = (256, 256)
    a_lr:float = 3e-4
    c_lr:float = 3e-4
    batch_size:int = 64
    alpha:float = 0.12
    adaptive_alpha:bool = True
    start_steps:int = 32
    train_every:int = 1

class SACtuner(Tuner):

    def __init__(self, optimize_mode='maximize', sac_args: dict[str, Any] | None = None):
        # super().__init__()
        self.optimize_mode = OptimizeMode(optimize_mode)

        self.sac_args = SACArguments(**(sac_args or {}))
        logger.info("SAC_tuner sac_args: %s", self.sac_args)

        self.model = None
        self.env = None

        self.state_dim = None
        self.action_dim = None
        self.replay_buffer = None

        self.action_adapter = None
        self.action_adapter_reverse = None

        self._generated_params = {}
        self._next_sates = {}
        self._current_states = {}

        self._state = None
        self._current_trial = 0

        self.received_rewards = {}

        self.space = None


    def update_search_space(self, search_space: SearchSpace) -> None:
        
        logger.info("Initializing SAC_tuner")
        newenv = SimpleENv(search_space)
        if self.env is not None:
            self.env.close()
            assert self.env.action_space.shape == newenv.action_space.shape, "action_space.shape should be equal"
            assert self.env.observation_space.shape == newenv.observation_space.shape, "observation_space.shape should be equal"

        self.env = newenv
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.model = SAC_Agent(state_dim=self.state_dim, action_dim=self.action_dim, 
                               gamma=self.sac_args.gamma, hid_shape=self.sac_args.hid_shape, 
                               a_lr=self.sac_args.a_lr, c_lr=self.sac_args.c_lr, 
                               batch_size=self.sac_args.batch_size, 
                               alpha=self.sac_args.alpha, adaptive_alpha=self.sac_args.adaptive_alpha)
        
        self.replay_buffer = RandomBuffer(state_dim=self.state_dim, action_dim=self.action_dim, Env_with_dead=True)

        self.action_adapter = Action_adapter(search_space)
        self.action_adapter_reverse = Action_adapter_reverse(search_space)

        self.space = search_space
        self._keys = list(search_space.keys())
        logger.info("Initialize SAC_tuner successfully")


    def generate_parameters(self, parameter_id: int, **kwargs) -> Parameters:
        
        assert self.env is not None and self.model is not None and self.replay_buffer is not None, "Please call update_search_space first"

        if self._current_trial < self.sac_args.start_steps:
            action = self.env.action_space.sample()
            act = self.action_adapter_reverse(action)
        else:
            act = self.model.select_action(self._state, 
                                           deterministic=False, with_logprob=False)
            action = self.action_adapter(act)

        self._generated_params[parameter_id] = action  # all generated parameters
        self._current_states[parameter_id] = self._state if self._state is not None else self.env.reset()
        self._next_sates[parameter_id] = act  # next_sates = act, generated parameters of current trial

        self._state = act
        self._current_trial += 1

        logger.info(f"generate_parameters: {parameter_id}")

        return {self._keys[i]: action[i] for i in range(len(action))}
    
    def receive_trial_result(self, parameter_id: int, parameters: Parameters, value: float, **kwargs) -> None:

        if parameter_id not in self._generated_params:
            logger.warning("parameter_id not in generated params")
            return
        
        reward = extract_scalar_reward(value)
        
        # reward = (reward+1000)/1000
        if self.optimize_mode == OptimizeMode.Minimize:
            reward = -reward

        self.received_rewards[parameter_id] = reward


    def trial_end(self, parameter_id: int, success: bool, **kwargs) -> None:
        
        if parameter_id not in self._generated_params:
            logger.warning("parameter_id not in generated params")
            return
        logger.info(f"trial_end: {parameter_id}, {success}")

        dead = not success
        # next_sates = action = generated parameters of current trial
        # logger.info(f'current_states, {self._current_states[parameter_id]}')
        # logger.info(f'next_states, {self._next_sates[parameter_id]}')
        # logger.info(f'received_rewards, {self.received_rewards[parameter_id]}')
        
        self.replay_buffer.add(self._current_states[parameter_id], self._next_sates[parameter_id], self.received_rewards[parameter_id], 
                               self._next_sates[parameter_id], dead)
        
        if self._current_trial > self.sac_args.start_steps//2 and self._current_trial % self.sac_args.train_every == 0:
            for _ in range(self.sac_args.train_every):
                self.model.train(self.replay_buffer)
            logger.info('Finished train new model')

        # if self._current_trial > self.sac_args.start_steps//2:
        #     self.model.train(self.replay_buffer)
        #     logger.info('Finished train new model')

        self._current_states.pop(parameter_id)
        self._next_sates.pop(parameter_id)
        self._generated_params.pop(parameter_id)
        self.received_rewards.pop(parameter_id)

    def import_data(self, data: list[TrialRecord]) -> None:
        logger.warning("import_data is not implemented")
        pass




        
    





