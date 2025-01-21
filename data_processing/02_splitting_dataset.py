import os
import pandas as pd
from sklearn.model_selection import train_test_split

# constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_SET_FOLDER = os.path.join(BASE_DIR, "./../data_set")
DATA_SET_FILE = "Training_set.csv"
TEST_SIZE = 0.3  # TRAIN_SIZE -> 0.7
GROUP_BY_COL = "label"
RANDOM_GENERATION_SEED = 33


# load the data
data = pd.read_csv(os.path.join(DATA_SET_FOLDER, DATA_SET_FILE))
whole_size = len(data)


# split the data based on each classes
train = []
test = []

for _, group in data.groupby(GROUP_BY_COL):
    train_subset, test_subset = train_test_split(
        group, test_size = TEST_SIZE, random_state = RANDOM_GENERATION_SEED
    )
    train.append(train_subset)
    test.append(test_subset)

train = pd.concat(train)
test = pd.concat(test)

# save the data (index=false to note save the idx of datafram as col)
train.to_csv(os.path.join(DATA_SET_FOLDER, "train_set.csv"), index=False)
test.to_csv(os.path.join(DATA_SET_FOLDER, "test_set.csv"), index=False)

# debug
print(f"Train set size: {len(train)}, expected: {round(whole_size * 0.7)}")
print(f"Test set size: {len(test)}, expected: {round(whole_size * 0.3)}")
