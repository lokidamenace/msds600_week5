import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv(r'C:\Users\peter\MSDS600\Week_5\churn_data_prepped.csv')
features = tpot_data.drop('Churn', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Churn'], random_state=42)

# Average CV score on the training set was: 0.7980652355729021
exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.25, min_samples_leaf=7, min_samples_split=6, n_estimators=100)

# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(results)
