import torch
import sys
sys.path.insert(0, '.')
from tests.common import FIXTURES_PATH

state_dict = torch.load(FIXTURES_PATH / 'ts_tests' / 'model.pt', map_location='cpu', weights_only=False)
for k in state_dict.keys():
    if 'bias' in k:
        print("Found bias:", k)
