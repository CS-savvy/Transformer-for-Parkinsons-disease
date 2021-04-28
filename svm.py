from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import datasets

import numpy as np
import pandas as pd
from pathlib import Path

this_dir = Path.cwd()

dataset_file = this_dir / "data/pd_speech_features.ods"
df = pd.read_excel(dataset_file, header=[0, 1], engine="odf")


print()

