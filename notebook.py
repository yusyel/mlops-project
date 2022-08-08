#!/usr/bin/env python
# coding: utf-8

# # 1. Data Data Preparation

# In[119]:


import pandas as pd
import numpy as np


# In[120]:


df = pd.read_csv('framingham.csv')


# In[121]:


df.columns = df.columns.str.lower()


# In[122]:


df.head().T


# In[123]:


df.dtypes


# In[124]:


df.nunique()


# In[125]:


df['male'] = df['male'].astype(str)
df['education'] = df['education'].astype(str)
df['currentsmoker'] = df['currentsmoker'].astype(str)
df['bpmeds'] = df['bpmeds'].astype(str)
df['prevalentstroke'] = df['prevalenthyp'].astype(str)
df['diabetes'] = df['diabetes'].astype(str)
df['prevalenthyp'] = df['prevalenthyp'].astype(str)


# In[126]:


df.nunique()


# In[127]:


df.dtypes


# ## 1.1 EDA for Missing Values

# In[128]:


numerical = ['age', 'cigsperday', 'totchol', 'sysbp',
             'diabp', 'bmi', 'heartrate', 'glucose']


# In[129]:


categorical = ['male', 'education', 'currentsmoker', 'bpmeds',
               'prevalentstroke', 'prevalenthyp', 'diabetes']


# In[130]:


df[categorical].head()


# In[131]:


df[categorical].isnull().sum()


# In[132]:


df.education = df.education.fillna(0)


# In[133]:


df.bpmeds = df.bpmeds.fillna(0)


# In[134]:


df[categorical].isnull().sum()


# In[135]:


df[numerical].isnull().sum()


# In[136]:


df.cigsperday = df.cigsperday.fillna(df.cigsperday.mean())


# In[137]:


df.heartrate = df.heartrate.fillna(df.heartrate.mean())


# categorical variable filled with 0, numerical varible filled mean values.

# In[138]:


df.glucose = df.glucose.fillna(df.glucose.mean())


# In[139]:


df.totchol = df.totchol.fillna(df.totchol.mean())


# In[140]:


df.bmi = df.bmi.fillna(df.bmi.mean())


# In[141]:


df.isnull().sum()


# In[142]:


df.head()


# In[143]:


df.columns


# In[144]:


df[numerical].head()


# ## 1.2 The Validation Frame

# In[145]:


from sklearn.model_selection import train_test_split


# In[146]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)
#df_test.to_csv('test.csv', index=False)
#df_test.to_parquet('output.parquet', index=False)
#df_train.to_parquet('train.parquet', index=False)


# In[147]:


df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)


# In[148]:


y_train = df_train.tenyearchd.values
y_val = df_val.tenyearchd.values
y_test = df_test.tenyearchd.values



# In[149]:


del df_train['tenyearchd']
del df_val['tenyearchd']
del df_test['tenyearchd']



# # 2. Exploratory Data Analysis

# ## 2.1 Exploratory Data Analysis For Numeric Variables

# In[150]:


import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[151]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
df_full_train.boxplot('age', 'tenyearchd', ax=ax[0])
df_full_train.boxplot('cigsperday', 'tenyearchd', ax=ax[1])


# Seems like people who 10 year risk of coronary heart disease slightly older than other class.
# For both class distribution of number of cigarettes that the person smoked on average in one day is  right skewed distribution.

# 

# In[152]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
df_full_train.boxplot('totchol', 'tenyearchd', ax=ax[0])
df_full_train.boxplot('sysbp', 'tenyearchd', ax=ax[1])


# Mean of **total cholesterol** level approximately 220-240 for both class and there is few outlier.
# Men of **Systolic blood pressure** for class 1 more than class 0. People who 10 year risk of CHD higher systolic blood pressure.

# In[153]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
df_full_train.boxplot('diabp', 'tenyearchd', ax=ax[0])
df_full_train.boxplot('bmi', 'tenyearchd', ax=ax[1])


# Mean of **diastolic blood pressure** 81 for class 0, 85 for class 1. People who 10 year risk of CHD higher diastolic blood pressure.But considering outlier of class 0 diastolic blood pressure is not enough the explanation risk of CHD.
# 
# 

# In[154]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
df_full_train.boxplot('heartrate', 'tenyearchd', ax=ax[0])
df_full_train.boxplot('glucose', 'tenyearchd', ax=ax[1])


# High **heart rate** doesn't cause ten year risk of CHD. But statistically has to be test. **Glucose level** contains many outlier observations for both class. It is not the only decisive factor.

# ## 2.2 Exploratory Data Analysis For Categorical Variable

# In[155]:


df_full_train[categorical].head()


# In[156]:


for cat in categorical:

    sns.catplot(data=df_full_train, kind='count', x=cat, hue='tenyearchd')


# Look like prevalent hypertension is an important variable. Interesting in here current smokers not considering as an important variable.

# ## 3.3) Exploratory Data Analysis For Correlated Data

# In[157]:


plt.figure(figsize=(15,10))  
sns.heatmap(df[numerical].corr(),annot=True,linewidths=.5)
plt.show()


# Systolic blood and diastolic blood pressure is highly correlated.

# In[158]:


x = df_train.diabp.sort_values()
y = df_train.sysbp.sort_values()
plt.plot(x,y)


# # MLflow Setup

# In[159]:


import mlflow
from mlflow import log_artifact

# In[160]:


from sklearn.feature_extraction import DictVectorizer

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope



# In[161]:


scope.int(hp.quniform('max_depth', 1, 2000, 1))


# In[162]:


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mlops-project")
log_artifacts(artifact_path="./services/prediction_service/mlruns")

# In[163]:


from sklearn.ensemble import RandomForestClassifier


# In[164]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score


# In[165]:


from sklearn.pipeline import make_pipeline


# In[166]:


train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)


val_dicts = df_val.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_val = dv.fit_transform(val_dicts)


# In[167]:


space = {'criterion' : hp.choice('criterion', ['entropy','gini']),
         'max_depth': scope.int(hp.quniform('max_depth', 1, 2000, 1)),
         'max_features' : hp.choice('max_features',['auto','sqrt','log2',None]),
         'min_samples_leaf' : hp.uniform('min_samples_leaf',0,0.5),
         'min_samples_split' : hp.uniform('min_samples_split',0,1),
         'n_estimators': scope.int(hp.quniform('n_estimators', 1, 2000, 1)),
         'max_leaf_nodes': scope.int(hp.quniform('max_leaf_nodes', 2, 1000,1)),
         'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
         'ccp_alpha': hp.uniform('ccp_alpha', 0,1)
    
}


# In[168]:


def objective(space):
    with mlflow.start_run():
        mlflow.set_tag("mlops", "model1")
        pipeline = make_pipeline(DictVectorizer(sparse=False), RandomForestClassifier(criterion= space['criterion'],
                                        max_depth = space['max_depth'],
                                        max_features = space['max_features'],
                                        min_samples_leaf = space['min_samples_leaf'],
                                        min_samples_split = space['min_samples_split'],
                                        n_estimators = space['n_estimators'],
                                        max_leaf_nodes = space['max_leaf_nodes'],
                                        n_jobs= -1))
        mlflow.log_params(space)
        pipeline.fit(train_dicts, y_train)
        y_pred_val = pipeline.predict_proba(val_dicts)[:,1]
        auc =roc_auc_score(y_val, y_pred_val)
        acc = accuracy_score(y_val, y_pred_val >= 0.55)
        f1 = f1_score(y_val, y_pred_val>=0.5, average = 'weighted')
        metrics = {"accuracy_score": acc , "auc" : auc, "f1":f1}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
    return {'loss' : auc, 'status' : STATUS_OK}



# In[169]:


rstate = np.random.default_rng(2)
params = fmin(fn = objective,
            space=space,
            algo= tpe.suggest,
            max_evals=100,
            rstate=rstate,
            trials = Trials()
           )

        


# # Model 2

# In[170]:


plt.figure(figsize=(15,10))  
sns.heatmap(df[numerical].corr(),annot=True,linewidths=.5)
plt.show()


# In[171]:


x = df_train.diabp.sort_values()
y = df_train.sysbp.sort_values()
plt.plot(x,y)


# In[172]:


df_train_model = df_train
df_val_model = df_val
df_test_model = df_test


# In[173]:


del df_train_model['diabp']
del df_val_model['diabp']
del df_test_model['diabp']


# In[174]:


train_dicts = df_train_model.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)


val_dicts = df_val_model.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_val = dv.fit_transform(val_dicts)


# In[175]:


space = {'criterion' : hp.choice('criterion', ['entropy','gini']),
         'max_depth': scope.int(hp.quniform('max_depth', 1, 2000, 1)),
         'max_features' : hp.choice('max_features',['auto','sqrt','log2',None]),
         'min_samples_leaf' : hp.uniform('min_samples_leaf',0,0.5),
         'min_samples_split' : hp.uniform('min_samples_split',0,1),
         'n_estimators': scope.int(hp.quniform('n_estimators', 1, 2000, 1)),
         'max_leaf_nodes': scope.int(hp.quniform('max_leaf_nodes', 2, 1000,1)),
         'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
         'ccp_alpha': hp.uniform('ccp_alpha', 0,1)
    
}


# In[176]:


def objective(space):
    with mlflow.start_run():
        mlflow.set_tag("mlops", "model2")
        pipeline = make_pipeline(DictVectorizer(sparse=False), RandomForestClassifier(criterion= space['criterion'],
                                        max_depth = space['max_depth'],
                                        max_features = space['max_features'],
                                        min_samples_leaf = space['min_samples_leaf'],
                                        min_samples_split = space['min_samples_split'],
                                        n_estimators = space['n_estimators'],
                                        max_leaf_nodes = space['max_leaf_nodes'],
                                        n_jobs= -1))
        mlflow.log_params(space)
        pipeline.fit(train_dicts, y_train)
        y_pred_val = pipeline.predict_proba(val_dicts)[:,1]
        auc =roc_auc_score(y_val, y_pred_val)
        acc = accuracy_score(y_val, y_pred_val >= 0.55)
        f1 = f1_score(y_val, y_pred_val>=0.5, average = 'weighted')
        metrics = {"accuracy_score": acc , "auc" : auc, "f1":f1}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
    return {'loss' : auc, 'status' : STATUS_OK}


# In[177]:


rstate = np.random.default_rng(2)
params = fmin(fn = objective,
            space=space,
            algo= tpe.suggest,
            max_evals=100,
            rstate=rstate,
            trials = Trials()
           )


# # Mlflow model Register

# In[178]:


from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
client.list_experiments()



# In[179]:


from mlflow.entities import ViewType

runs = client.search_runs(
    experiment_ids='1',
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.auc DESC"])


# In[180]:


for run in runs:
    print(
        f"run id: {run.info.run_id}, accuracy_score: {run.data.metrics['accuracy_score']:.4f}, auc_score: {run.data.metrics['auc']:.4f}")


# In[181]:


best = client.search_runs(
    experiment_ids='1',
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.auc DESC"])[0]


# In[182]:


best.info.run_id


# In[183]:


best_model = best.info.run_id


# In[184]:


mlflow.register_model(
    model_uri=f"runs:/{best_model}/model",
    name = "CHD_risk_model"
)


# # Notebook Test:

# In[185]:


import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[186]:


df = pd.read_csv('framingham.csv')
df.columns = df.columns.str.lower()
df['male'] = df['male'].astype(str)
df['education'] = df['education'].astype(str)
df['currentsmoker'] = df['currentsmoker'].astype(str)
df['bpmeds'] = df['bpmeds'].astype(str)
df['prevalentstroke'] = df['prevalenthyp'].astype(str)
df['diabetes'] = df['diabetes'].astype(str)
df['prevalenthyp'] = df['prevalenthyp'].astype(str)

numerical = ['age', 'cigsperday', 'totchol', 'sysbp',
             'diabp', 'bmi', 'heartrate', 'glucose']


categorical = ['male', 'education', 'currentsmoker', 'bpmeds',
               'prevalentstroke', 'prevalenthyp', 'diabetes']

df.cigsperday = df.cigsperday.fillna(df.cigsperday.mean())
df.heartrate = df.heartrate.fillna(df.heartrate.mean())
df.glucose = df.glucose.fillna(df.glucose.mean())
df.totchol = df.totchol.fillna(df.totchol.mean())
df.bmi = df.bmi.fillna(df.bmi.mean())


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)


df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

y_train = df_train.tenyearchd.values
y_val = df_val.tenyearchd.values
y_test = df_test.tenyearchd.values


del df_train['tenyearchd']
del df_val['tenyearchd']
del df_test['tenyearchd']




# In[187]:


train_dicts = df_train.to_dict(orient = 'records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_train.shape


# In[188]:


rf = RandomForestClassifier(            criterion = "gini",
                                        max_depth = 1983,
                                        max_features = None,
                                        min_samples_leaf = 0.26498195795743534,
                                        min_samples_split = 0.033261893714441826,
                                        n_estimators = 1455,
                                        max_leaf_nodes =172)
rf.fit(X_train, y_train)
print(X_train.shape)


# In[189]:


test= {'male': '0',
 'age': 54,
 'education': '1.0',
 'currentsmoker': '0',
 'cigsperday': 0.0,
 'bpmeds': '0.0',
 'prevalentstroke': '1',
 'prevalenthyp': '1',
 'diabetes': '0',
 'totchol': 315.0,
 'sysbp': 176.0,
 'diabp': 87.0,
 'bmi': 29.23,
 'heartrate': 82.0,
 'glucose': 72.0}


# In[190]:


x = dv.transform(test)
print(x.shape)
rf.predict_proba(x)[0,1]


# In[191]:


import mlflow
import mlflow.sklearn


# In[192]:


logged_model = f"./mlruns/1/{best_model}/artifacts/model"
model = mlflow.sklearn.load_model(logged_model)


# In[ ]:


print(model)
model.predict_proba(test)


# In[ ]:


model.predict_proba(test)[0,1]


# In[ ]:




