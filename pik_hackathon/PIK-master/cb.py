import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from preprocess import preprocess_train, preprocess, FEATURES, \
    CATEGORICAL_FEATURES, TEST_FEATURES, CATEGORICAL_TEST_FEATURES_IDX


# DATA
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X, y = preprocess_train(train,
                        categotical_features=CATEGORICAL_FEATURES,
                        features=TEST_FEATURES)

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)


print "N features: {}".format(X.shape[1])

# Training simple model
from catboost import CatBoostRegressor


cbr = CatBoostRegressor(iterations=10000,
    logging_level='Silent',
    depth=10,
    #task_type='GPU',
    )


scorer = make_scorer(lambda a, b: mean_squared_error(a, b)**.5)
scores = cross_val_score(cbr, X, y, cv=5, scoring=scorer, verbose=1)

print "Mean result: {}".format(np.mean(scores))