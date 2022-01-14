import pandas as pd

def make_submission(ids, predictions):
    df = pd.concat([ids, pd.Series(predictions)], axis=1)
    return df.rename(columns={0: 'value'})


import cPickle as pickle

def save(obj, path):
    with open(path, 'w') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'r') as f:
        return pickle.load(f)