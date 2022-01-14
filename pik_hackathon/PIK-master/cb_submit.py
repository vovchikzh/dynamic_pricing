import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from preprocess import preprocess_train, preprocess, FEATURES, \
    CATEGORICAL_FEATURES, TEST_FEATURES, CATEGORICAL_TEST_FEATURES_IDX
from utils import make_submission

# DATA
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X, y = preprocess_train(train,
                        categotical_features=CATEGORICAL_FEATURES,
                        features=TEST_FEATURES)

X, y = shuffle(X, y)


# Training simple model
from catboost import CatBoostRegressor


cbr = CatBoostRegressor(iterations=10000,
    logging_level='Silent',
    depth=10,
    # task_type='GPU',
    )


# scorer = make_scorer(lambda a, b: mean_squared_error(a, b)**.5)
# scores = cross_val_score(cbr, X, y, cv=5, scoring=scorer, verbose=1)
#
# print "Mean result: {}".format(np.mean(scores))
cbr.fit(X, y)

print "Training finished"

mean_y = np.mean(y)
min_y = np.min(y)

test = preprocess(test, CATEGORICAL_FEATURES)
X_test = test[TEST_FEATURES].values
predictions = cbr.predict(X_test)

df = pd.concat([test.id, pd.Series(predictions)], axis=1)
df = df.rename(columns={0: 'value'})
df.to_csv("catboost_d10_i10000_f1.csv", index=False)

# Tweaked
df_nonneg_min = df.copy()
df_nonneg_min[df_nonneg_min < 0] = min_y
df.to_csv("catboost_d10_i10000_nnmin_f1.csv", index=False)

df_nonneg_mean = df.copy()
df_nonneg_mean[df_nonneg_mean < 0] = mean_y
df.to_csv("catboost_d10_i10000_nnmean_f1.csv", index=False)