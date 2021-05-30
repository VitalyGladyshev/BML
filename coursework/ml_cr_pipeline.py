"""
Курсовой проект по BML Гладышева ВВ
ML задача предсказания дефолтов по кредитам

Файл содержит pipeline данных для получения и оценки модели
"""

import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix


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


class CatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        column = x.copy()
        if self.key == "Years in current job":
            map_jb = {
                '< 1 year': 0.5,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10
            }
            column = column.map(map_jb)
        elif self.key == "Home Ownership":
            map_ho = {
                'Have Mortgage': 2,
                'Home Mortgage': 2,
                'Own Home': 3,
                'Rent': 1
            }
            column = column.map(map_ho)
        elif self.key == "Term":
            map_t = {
                'Short Term': 0,
                'Long Term': 1
            }
            column = column.map(map_t)
            column = column.astype(np.int64)
        elif self.key == "Purpose":
            map_p = {
                'business loan': 1,
                'buy a car': 2,
                'buy house': 3,
                'debt consolidation': 4,
                'educational expenses': 5,
                'home improvements': 6,
                'major purchase': 7,
                'medical bills': 8,
                'moving': 9,
                'other': 10,
                'renewable energy': 11,
                'small business': 12,
                'take a trip': 13,
                'vacation': 14,
                'wedding': 15
            }
            column = column.map(map_p)
        column.replace([np.inf, -np.inf], np.nan, inplace=True)
        column[column.isnull()] = 0
        return pd.DataFrame(column)


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
data_dir = "./data/"
models_dir = "./models/"
data = pd.read_csv(data_dir + "pds_project_train.csv")

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
X_train, X_valid, y_train, y_valid = train_test_split(data.drop(["Credit Default", "Id"], axis=1),
                                                      data["Credit Default"],
                                                      test_size=0.2, random_state=42)

feat_eng_columns = ["Maximum Open Credit", "Annual Income", "Current Loan Amount",
                    "Current Credit Balance", "Monthly Debt", "Credit Score"]
other_columns = list(set(data.columns) - set(cat_list) - set(feat_eng_columns) - {'Id', "Credit Default"})

final_transformers = list()

for feat_eng_col in feat_eng_columns:
    feat_eng_transformer = Pipeline([('feat_eng', FeatureCreator(key=feat_eng_col))])
    final_transformers.append((feat_eng_col, feat_eng_transformer))

for cat_col in cat_list:
    cat_transformer = Pipeline([('selector', ColumnSelector(key=cat_col)),
                                ('ohe', CatEncoder(key=cat_col))
                                ])
    final_transformers.append((cat_col, cat_transformer))

for other_col in other_columns:
    other_transformer = Pipeline([
        ('selector', NumberSelector(key=other_col))
    ])
    final_transformers.append((other_col, other_transformer))

feats = FeatureUnion(final_transformers)
feature_processing = Pipeline([('feats', feats)])

pipeline = Pipeline([('features', feats),
                     ('gbc', GradientBoostingClassifier(random_state=42))])

# Обучаем модель
pipeline.fit(X_train, y_train)

# Сохраняем модель
with open(models_dir + "model_cr_gbc.dill", "wb") as f:
    dill.dump(pipeline, f)

# Оценка модели
cl_met = ['Best Threshold', 'F-Score', 'Precision',
          'Recall', 'roc_auc_s', 'log_loss_s', 'TPR', 'FPR', 'TNR', "TN", "FN", "TP", "FP"]

res_tab = pd.DataFrame(columns=cl_met)

predictions = pipeline.predict_proba(X_valid)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_valid, predictions)
f_score = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(f_score)
print('Best Threshold=%f, F-Score=%.3f, Precision=%.3f, Recall=%.3f' % (thresholds[ix],
                                                                        f_score[ix],
                                                                        precision[ix],
                                                                        recall[ix]))

r_auc = roc_auc_score(y_true=y_valid, y_score=predictions)
l_los = log_loss(y_true=y_valid, y_pred=predictions)

print("roc auc score: {}".format(r_auc))
print("log loss score: {}".format(l_los))

cnf_matrix = confusion_matrix(y_valid, predictions > thresholds[ix])

TN = cnf_matrix[0][0]
FN = cnf_matrix[1][0]
TP = cnf_matrix[1][1]
FP = cnf_matrix[0][1]

TPR = TP/(TP+FN)
FPR = FP/(FP+TN)
TNR = TN/(FP+TN)

res_tab.loc['GradientBoostingClassifier', :] = [thresholds[ix],
                                                f_score[ix],
                                                precision[ix],
                                                recall[ix],
                                                r_auc, l_los,
                                                TPR, FPR, TNR,
                                                TN, FN, TP, FP]

print(res_tab.sort_values('roc_auc_s', ascending=False))
