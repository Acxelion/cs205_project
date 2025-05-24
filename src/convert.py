from musegan.data import load_data
import numpy as np

import os
import os.path
from pathlib import Path

with open("training_data.npz", "wb") as f:
  sparseFormat = load_data("npz","./train_x_lpd_5_phr.npz")
  print(type(sparseFormat), sparseFormat.shape)

  np.save(f, sparseFormat)
  print("Success")