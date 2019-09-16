# -*- coding: utf-8 -*-

#########################Bank Marketing Data Set
#https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

#Attribute Information:
#
#Input variables:
## bank client data:
#1 - age (numeric)
#2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
#3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
#4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
#5 - default: has credit in default? (categorical: 'no','yes','unknown')
#6 - housing: has housing loan? (categorical: 'no','yes','unknown')
#7 - loan: has personal loan? (categorical: 'no','yes','unknown')
## related with the last contact of the current campaign:
#8 - contact: contact communication type (categorical: 'cellular','telephone')
#9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
#10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
#11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
## other attributes:
#12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#14 - previous: number of contacts performed before this campaign and for this client (numeric)
#15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
## social and economic context attributes
#16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#17 - cons.price.idx: consumer price index - monthly indicator (numeric)
#18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
#19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
#20 - nr.employed: number of employees - quarterly indicator (numeric)
#
#Output variable (desired target):
#21 - y - has the client subscribed a term deposit? (binary: 'yes','no')'

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import  TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import randint as sp_randint
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

#%%
class ColumnExtractor(TransformerMixin):
    """Transformador que selecciona columnas de un dataframe"""
    def __init__(self, columns):
        self.columns = columns
        
    def transform(self, X, **transform_params):
        return X[self.columns].as_matrix()
        
    def fit(self, X, y=None, **fit_params):
        return self
    
class ClassicMultipleBinarizer(preprocessing.MultiLabelBinarizer):
    def fit(self, X, y=None):
        super(ClassicMultipleBinarizer, self).fit(X)
        
    def transform(self, X, y=None):
        return super(ClassicMultipleBinarizer, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(ClassicMultipleBinarizer, self).fit(X).transform(X)

def missing_values(df):
    print('Number of missing values for feature:')
    print('From {} values'.format(len(df)))
    drop_features=list()
    cont=0
    for col in df:
        nu_mv=len(df)-df[col].notnull().sum()
        if nu_mv>=1:
            per_missing=(nu_mv*100)/len(df)
            print('{}:  {} missing values {:.2f}% of feature data'.format( col,nu_mv,per_missing))
            if per_missing>40.0:
                drop_features.append(col)
            cont +=nu_mv
    if cont==0:
            print('...No missing values')
    
    for col in drop_features:
        df=df.drop([col],axis=1)
        print('\nDrop column {} with missing values > 40%'.format(col))
    return df
#%%
print('---------- Bank Marketing Data Set ----------')
print('\n')
print('Abstract: The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ("yes") or not ("no") subscribed.  ')

print('Download data: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing ')

print('\n')
print('Reading data to training...')
df=pd.read_csv('data_train/train_data.csv') 

#%%
print('\n')
print('Size of data: ')
print('{} Examples'.format(df.shape[0]))
print('{} Features'.format(df.shape[1]))

#%%
print('\n')
print('Data: ')
print(df.head())
#%%
y=df['y']
df=df.drop(['y'],axis=1)
#%%

print('\n')
print('Missing data??')
df=missing_values(df)

#%%

print('\n')
print('Drop duplicated rows ...')
df=df.drop_duplicates()
#%%
print('\n')
print('Targets labeled with "no" are labeled with "0"')
print('Targets labeled with "yes" are labeled with "1"')
y=y.replace({"no":0, "yes":1})
#%%
num_positives=y.value_counts()[1]
num_negatives=y.value_counts()[0]
print('\n')
print('Examples positives: {} '.format(num_positives))
print('Examples negatives: {}'.format(num_negatives))
print('¡¡¡¡ We have umbalanced data !!!!')

#%%
print('\n')
print('Undersample majority class for training data... ')

from sklearn.utils import resample
X = pd.concat([pd.DataFrame(df), pd.DataFrame(y)], axis=1)
# separate minority and majority classes
Y = X[X.y==1]
N = X[X.y==0]
#%% Removing some observations of the majority class
N_downsampled = resample(N, replace=True, # sample without replacement
                          n_samples=len(Y), # match minority n
                          random_state=42) # reproducible results
# combine minority and downsampled majority
downsampled = pd.concat([Y, N_downsampled])
print('Now, Balanced classes to train and validate the model.')
# check new class counts
print(downsampled.y.value_counts())

y=downsampled.y
df=downsampled.drop('y',axis=1)

#%%
#now, separate the features by data_type
numeric_columns = df.select_dtypes([np.number]).columns
categorical_columns = df.select_dtypes([object]).columns
#%% pipeline by numerical features

pipeline_numerical = Pipeline([
    ('selector_numeric ', ColumnExtractor(columns=numeric_columns)),
    ('imputer_missing_values', SimpleImputer(missing_values=np.nan, strategy='mean'))
])
#%% pipeline by categorical/ordinal  features
    
pipeline_categorical = Pipeline([
    ('selector_categorical', ColumnExtractor(columns=categorical_columns)),
    ('imputer_missing_values', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('ClassicMultipleBinarizer', ClassicMultipleBinarizer())
])
#%% union of pipelines
    
pipeline_features = FeatureUnion([
    ('pipeline_numerical', pipeline_numerical),
    ('pipeline_categorical', pipeline_categorical)
])
#%% pipeline union

pipeline_union=Pipeline([
        ('preprocessed_data', pipeline_features),
        ('feature_selection', VarianceThreshold()),
        ('feature_extraction', PCA(n_components = 20)),#11)),
        ('scaler', StandardScaler())
        ])

data_procesada = pipeline_union.fit_transform(df)

#%% use  RandomizedSearchCV and select the best estimator

param_dist_random = {
    "max_depth": [3, None],
    "max_features": sp_randint(1, 20),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "n_estimators": np.linspace(10,100,10).astype(int),
}

RandomForestC_10 = RandomizedSearchCV(estimator=RandomForestClassifier(), 
                    param_distributions=param_dist_random,
                   scoring="roc_auc", n_jobs=-1, n_iter=10)

#%%   final pipeline with the estimator
pipeline_estimator = Pipeline(
    [
     ("pipeline_union", pipeline_union),
     ("estimator_RandomizedSearchCV", RandomForestC_10)
    ])

#%%
print('\n')
print('Training model......')
pipeline_estimator.fit(df, y)

#%%
print('\n')
print('Ready! ')
#%%
print('Now.... Reading data for testing!!')
df_test=pd.read_csv('data_test/test_data.csv') 

#%% separate features and target feature
y_test=df_test['y']
df_test=df_test.drop(['y'],axis=1)

#%% encoder the target feature
y_test=y_test.replace({"no":0, "yes":1})

#%%
print('Testing model...')
predictions=pipeline_estimator.predict(df_test)

#%%
print('\n')
print('Comparison of results...')
comparison_results=pd.DataFrame(y_test)
comparison_results['predictions']=predictions
comparison_results=comparison_results.sort_values(by='y',ascending=False)

#%% calculate the number of positive and negative examples of data_test
num_negatives=y_test.value_counts()[0]
num_positives=y_test.value_counts()[1]

#%%

print('Results: ')
TP=(comparison_results.query('y==predictions and y==1')).shape[0]
TN=(comparison_results.query('y==predictions and y==0')).shape[0]
FP=(comparison_results.query('y!=predictions and y==0')).shape[0]
FN=(comparison_results.query('y!=predictions and y==1')).shape[0]
ACC=(TP+TN)/y_test.shape[0]
SEN=TP/num_positives
ESP=TN/num_negatives
PREC=TP/(TP+FP)


print('accuracy_score: ', accuracy_score(y_test,predictions))
print('recall_score: ', recall_score(y_test,predictions))

#%%
#confusion matrix
CONFUSION_MATRIX=pd.DataFrame([TP,FN], columns=['POSITIVE CLASS'])
CONFUSION_MATRIX['NEGATIVE CLASS']=[FP,TN]
new_index=['Prediction POS','Prediction NEG']
CONFUSION_MATRIX=CONFUSION_MATRIX.set_index([new_index])
print('\nConfusion Matrix:')
print(CONFUSION_MATRIX)

#PLOT ROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, predictions)
auc = roc_auc_score(y_test, predictions)
    
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print('\n')
print('ROC curve (area = {:.2f})'.format(auc))

comparison_results.to_csv ('results/results.csv', index = None, header=True)


