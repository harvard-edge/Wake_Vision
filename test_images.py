from wake_vision_loader import get_wake_vision
from vww_loader import get_vww

import tensorflow as tf
import matplotlib.pyplot as plt

train, val, test = get_vww(1)

samples = test.take(1)
for sample in samples:
    plt.imshow(sample[0].numpy().squeeze())
    plt.savefig(f'test_vww.png')
    break