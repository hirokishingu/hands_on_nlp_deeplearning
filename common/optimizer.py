import sys
sys.path.append('..')
from common.layers import *

class SDG:
    def __init__(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr = * grads[i]
            