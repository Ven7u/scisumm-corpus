import os
import pandas as pd
from sklearn.externals import joblib

class cache_df:

    def __init__(self, cache_folder="../cache/"):
        self.cache_folder = cache_folder
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

    def save_df(self, df, file_name, **kwargs):
        df.to_csv(self.cache_folder + file_name, **kwargs)

    def load_df(self, file_name, **kwargs):
        return pd.read_csv(self.cache_folder + file_name, **kwargs)

    def check(self, file_name):
        return os.path.exists(self.cache_folder + file_name)


    def save_scikit_model(self, clf, name):
        joblib.dump(clf, self.cache_folder+name)

    def load_scikit_model(self, name):
        return joblib.load(self.cache_folder+name)