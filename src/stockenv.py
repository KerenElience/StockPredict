import gymnasium as gym
from utils.common import np


class ObserveSpace():
    def __init__(self, dim):
        self.shape = (dim, )

#定义动作空间(Agent可选执行的动作)
class ActionSpace():
    """
    agent action: bought, hold, sold out
    """
    def __init__(self, n):
        #可选动作总数
        self.n = n

    def seed(self, seed):
        pass

    def sample(self, ):
        return np.random.randint(0, self.n - 1)

class StockEnv():
    def __init__(self):
        pass