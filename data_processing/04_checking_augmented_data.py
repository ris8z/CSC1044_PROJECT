import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SET_FOLDER = os.path.join(BASE_DIR, "./../data_set")
DATA_SET_FILE = "train_set_augmented.csv"

d = {}
with open(os.path.join(DATA_SET_FOLDER, DATA_SET_FILE), "r") as f:
    f.readline()  # drop first line
    for line in f:
        tmp = line.split(",")[-1].strip()
        d[tmp] = d.get(tmp, 0) + 1
    print(d)

print("n of lables", len(d))
print("min", min(d.values()))
print("max", max(d.values()))
print("average", sum(d.values()) / len(d.values()))
print("all values", sorted(d.values()))
print("number of images", sum(d.values()))
