import os
import numpy as np

npz = np.load("outputs/maila_depths.npz")
depths = npz["depths"]

npy = np.load("outputs/maila_depths.npy")
pass