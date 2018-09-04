from functools import reduce
import pandas as pd

path = "../data/"
train_path = path + "train-jpg/"
train = pd.read_csv(path + "train_v2.csv")

tags = list(reduce(lambda x, y: x | set(y.split()), train['tags'], set()))

for tag in tags:
    train[tag] = train['tags'].apply(lambda x: tag in x.split())

train = train.drop(labels='tags', axis=1)
