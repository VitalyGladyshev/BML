"""
Курсовой проект по BML Гладышева ВВ
ML задача предсказания дефолтов по кредитам

Файл содержит pipeline данных
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x_inc, y=None):
        return self

    def transform(self, x_inc):
        x_inc[self.key] = x_inc[self.key].fillna(0)
        return x_inc[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x_inc, y=None):
        return self

    def transform(self, x_inc):
        x_inc[self.key] = x_inc[self.key].fillna(0)
        return x_inc[[self.key]]


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, x_inc, y=None):
        df = x_inc.copy()
        self.columns = [col for col in pd.get_dummies(df, prefix=self.key).columns]
        return self

    def transform(self, x_inc):
        df = x_inc.copy()
        df = pd.get_dummies(df, prefix=self.key)
        test_columns = [col for col in df.columns]
        for col_ in test_columns:
            if col_ not in self.columns:
                df[col_] = 0
        return df[self.columns]


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, x_inc, y=None):
        #         print(f"fit: key: {self.key} columns: {x_inc.columns}")
        self.columns.clear()
        return self

    def transform(self, x_inc):
        #         print(f"transform: key: {self.key} columns: {x_inc.columns}")
        self.columns.clear()
        df = x_inc.copy()
        df[self.key] = df[self.key].fillna(0)
        df[self.key + '_log'] = np.log(df[self.key] + 1.1)
        self.columns.append(self.key + '_log')

        for i2, col2 in enumerate(feat_eng_columns):
            df[col2] = df[col2].fillna(0)
            df['%s_%s_4' % (self.key, col2)] = df[self.key] * df[col2]
            self.columns.append('%s_%s_4' % (self.key, col2))

            df['%s_%s_44' % (self.key, col2)] = df[self.key] * np.log(df[col2] + 1)
            self.columns.append('%s_%s_44' % (self.key, col2))

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns.to_list():
            df.loc[df[col].isnull(), col] = 0

        return df[self.columns]


# Загружаем и готовим данные
data = pd.read_csv("./data/pds_project_train.csv")

data['Id'] = np.arange(0, len(data))
cat_list = data.select_dtypes(include='object').columns

data.loc[(data['Bankruptcies'].isnull() & data['Months since last delinquent']), 'Bankruptcies'] = 1
data.loc[(data['Bankruptcies'].isnull() & (data['Months since last delinquent'] == 0)), 'Bankruptcies'] = 0

tmp = data.loc[data['Months since last delinquent'] > 0].copy()

tmp_lst = tmp.groupby(['Number of Credit Problems'], as_index=False) \
    .agg({'Months since last delinquent': 'mean'})['Months since last delinquent'].tolist()

for i in range(0, 8):
    data.loc[(((data['Months since last delinquent'].isnull()) |
               (data['Months since last delinquent'] == 0)) &
              (data['Number of Credit Problems'] == i)), ['Months since last delinquent']] = tmp_lst[i]

data.loc[data['Id'] == 6472, 'Annual Income'] = 1014934
data.loc[data['Id'] == 44, 'Maximum Open Credit'] = 3800528
data.loc[data['Id'] == 617, 'Maximum Open Credit'] = 1304726
data.loc[data['Id'] == 2617, 'Maximum Open Credit'] = 2651287

# Реализуем Pipeline
x_inc_train, x_inc_valid, y_train, y_valid = train_test_split(data.drop(["Credit Default", "Id"], axis=1),
                                                      data["Credit Default"],
                                                      test_size=0.2, random_state=42)

feat_eng_columns = ["Maximum Open Credit", "Annual Income", "Current Loan Amount",
                    "Current Credit Balance", "Monthly Debt", "Credit Score"]
other_columns = list(set(data.columns) - set(cat_list) - set(feat_eng_columns) - {'Id', "Credit Default"})

print(f"{cat_list}\n{feat_eng_columns}\n{other_columns}")
