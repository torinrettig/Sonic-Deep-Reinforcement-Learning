# Based on Alexandre Borghi's Sonic contest code: https://github.com/aborghi/retro_contest_agent

import numpy as np
import gym
import gym_remote.client as grc
from baselines.common.atari_wrappers import FrameStack

import cv2
cv2.ocl.setUseOpenCL(False)

from retro_contest.local import make
from retro import make as make_retro

class WarpFrame96(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 96x96."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def make_custom(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame96(env)
    if stack:
        env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env

def make_env(env_idx):
    """
    Create an environment with some standard wrappers.

    dicts = [
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'EmeraldHillZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'ChemicalPlantZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'ChemicalPlantZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'OilOceanZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'OilOceanZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act1'}
    ]
    """

    dicts = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}


    ]
    # Make the environment
    print(dicts[env_idx]['game'], dicts[env_idx]['state'], flush=True)
    #record_path = "./records/" + dicts[env_idx]['state']
    env = make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'], bk2dir="./bk2dir/t1")#, bk2dir="./records")#record='/tmp')

    # Use Monitor to record
    env = gym.wrappers.Monitor(env, "./recording/m1vid", force=True)

    # Build the actions array, 
    env = SonicDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = WarpFrame96(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env


def make_test():
    """
    Create an environment with some standard wrappers.
    
    dicts = [
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'EmeraldHillZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'ChemicalPlantZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'ChemicalPlantZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'OilOceanZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'OilOceanZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act1'}
    ]
    """
    # Here we add record because we want to output a video
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

    # Build the actions array, 
    env = SonicDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = WarpFrame96(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env


def make_train_0():
    return make_env(0)

def make_train_1():
    return make_env(1)

def make_train_2():
    return make_env(2)

def make_train_3():
    return make_env(3)

def make_train_4():
    return make_env(4)

def make_train_5():
    return make_env(5)

def make_train_6():
    return make_env(6)

def make_train_7():
    return make_env(7)

def make_train_8():
    return make_env(8)

def make_train_9():
    return make_env(9)

def make_train_10():
    return make_env(10)

def make_train_11():
    return make_env(11)

def make_train_12():
    return make_env(12)