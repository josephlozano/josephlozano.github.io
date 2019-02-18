---
title: "Loan Prediction - Feature Engineering & Modeling"
layout: post
date: 2018-12-14
tag: posting
projects: true
hidden: true # don't count this post in blog pagination
description: "Binary Classification Project"
category: blog
author: josephlozano
externalLink: false
---

# Feature Engineering & Modeling


```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import re
from statistics import mean

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
```

**Function to assess the models using accuracy, precision, recall, and F1 score.**


```python
def joeAssess(X_train, y_train, X_test, y_test, models):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    names = []
    
    for key, algo in models.items():
        names.append(key)
        algo.fit(X_train, y_train)
        predicted_y = algo.predict(X_test)
        
        first_score = accuracy_score(y_test, predicted_y)
        second_score = precision_score(y_test, predicted_y)
        third_score = recall_score(y_test, predicted_y)
        fourth_score = f1_score(y_test, predicted_y)
        
        accuracy.append(first_score)
        precision.append(second_score)
        recall.append(third_score)
        f1.append(fourth_score)
        
    metrics = pd.DataFrame(columns = ['Accuracy','Precision','Recall','F1'],
                       index= names)
    
    metrics['Accuracy'] = accuracy
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1'] = f1
    
    metrics.replace(1,0,inplace = True)
    
    return metrics.sort_values('F1', ascending = False)
```

**Models tested**


```python
g = GaussianNB()
b = BernoulliNB()
k = KNeighborsClassifier()
d = DecisionTreeClassifier()
r = RandomForestClassifier(n_estimators=100)
gbc = GradientBoostingClassifier(n_estimators=100)
```


```python
models = {
  "GaussianNB": g,
  "Bernoulli NB": b,
  "KNeighbors": k,
  "DescisionT": d,
  "RandomF": r,
  "GradientB": gbc,
}
```


```python
train = pd.read_csv('train_cleaned.csv')
```


```python
definitions = {'Loan ID': 'A unique Identifier for the loan information.',
               'Customer ID': 'A unique identifier for the customer. Customers may have more than one loan.',
               'Loan Status': 'A categorical variable indicating if the loan was paid back or defaulted.',
               'Current Loan Amount': 'This is the loan amount that was either completely paid off, or the amount that was defaulted.',
               'Term': 'A categorical variable indicating if it is a short term or long term loan.',
               'Credit Score': 'A value between 0 and 800 indicating the riskiness of the borrowers credit history.',
               'Years in current job': 'A categorical variable indicating how many years the customer has been in their current job.',
               'Home Ownership': 'Categorical variable indicating home ownership. Values are "Rent", "Home Mortgage", and "Own". If the value is OWN, then the customer is a home owner with no mortgage',
               'Annual Income': "The customer's annual income",
               'Purpose': 'A description of the purpose of the loan.',
               'Monthly Debt': "The customer's monthly payment for their existing loans",
               'Years of Credit History': "The years since the first entry in the customer’s credit history",
               'Months since last delinquent': 'Months since the last loan delinquent payment',
               'Number of Open Accounts': 'The total number of open credit cards',
               'Number of Credit Problems': 'The number of credit problems in the customer records.',
               'Current Credit Balance': 'The current total debt for the customer',
               'Maximum Open Credit': 'The maximum credit limit for all credit sources.',
               'Bankruptcies': 'The total number of bankruptcies.',
               'Tax Liens': 'The number of tax liens.'}
```


```python
def define(column_name):
    """Returns definition of column name if name is in definitions"""
    try:
        print(definitions[column_name])
    except KeyError:
        print("'" + column_name + "'",'not found in definitions.')
```

## Assigning credit scores to known ranges

**Credit Score Bands/Ranges**

<img style="float: left;" src="https://www.experian.com/blogs/ask-experian/wp-content/uploads/2016/10/experian-good-score-ranges.png">


```python
def credit_score_bands(x):
    if x >= 800 and x <= 850:
        return 5
    elif x >= 740:
        return 4
    elif x >= 670:
        return 3
    elif x >= 580:
        return 2
    elif x >= 300:
        return 1
    else:
        return np.nan
```


```python
train['Credit Score Ranges'] = train['Credit Score'].apply(credit_score_bands)
```


```python
sum(train['Credit Score Ranges'].isnull())
```




    0



**Most of the credit scores land in the 3rd and 4th band**


```python
sns.catplot(data=train, x='Credit Score Ranges', kind='count')
```




    <seaborn.axisgrid.FacetGrid at 0x10ad24eb8>




![png](/assets/images/loan_prediction_feature_engineering/output_17_1.png)


**Creating a new feature where the missing values of 'Months since last delinquent' signify the loan was never delinquent**


```python
train['Was Deliquent'] = train['Months since last delinquent'].apply(lambda x: 0 if pd.isnull(x) else 1)
```


```python
train['Was Deliquent'].value_counts()
```




    1    58738
    0    52353
    Name: Was Deliquent, dtype: int64




```python
train.drop('Months since last delinquent',axis=1,inplace=True)
```

**Create a feature to check current credit usage**


```python
train['Current Credit Usage'] = train['Current Credit Balance'] / train['Maximum Open Credit']
```

**Feature to check months left at the current monthly loan pay down. Replaces 'Monthly Debt' of 0 to 1 to avoid dividing by zero. This will cause the months left to be displayed to be as if the loan was paid off 1 dollar a day if 'Monthly Debt' is zero.**


```python
train['Months left'] =  train['Current Loan Amount'] / train['Monthly Debt'].replace({ 0 : 1 })
```


```python
train['Months left'].describe()
```




    count    111091.000000
    mean         30.605453
    std         363.427566
    min           0.353682
    25%           9.359466
    50%          14.706457
    75%          22.869155
    max       35459.000000
    Name: Months left, dtype: float64




```python
train['Months left'].fillna(0,inplace=True)
```


```python
sns.boxplot(x="Months left", data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1635dac8>




![png](/assets/images/loan_prediction_feature_engineering/output_28_1.png)


**Creates a monthly debt to income ratio to see how much of their monthly income is used for loan paydown. Higher ratio should indicate a higher risk of defaulting on loan.**


```python
train['Monthly Debt/Income'] = train['Monthly Debt'] / ( train['Annual Income'] / 12 )
```


```python
train['Monthly Debt/Income'].describe()
```




    count    111091.000000
    mean          0.166839
    std           0.078587
    min           0.000000
    25%           0.108000
    50%           0.162999
    75%           0.221001
    max           0.400001
    Name: Monthly Debt/Income, dtype: float64




```python
sns.distplot(train['Monthly Debt/Income'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10f329240>




![png](/assets/images/loan_prediction_feature_engineering/output_32_1.png)


**The effect of 'Term' on 'Loan Status' seems to depend on the 'Current Loan Amount" so an interaction term should help the model.**


```python
sns.catplot(x="Term", y="Current Loan Amount", hue="Loan Status", data=train, kind="bar")
```




    <seaborn.axisgrid.FacetGrid at 0x10beec240>




![png](/assets/images/loan_prediction_feature_engineering/output_34_1.png)


**Also creating an interaction term between 'Term' and 'Annual Income'.**


```python
sns.catplot(x="Term", y="Annual Income", hue="Loan Status", data=train, kind="bar")
```




    <seaborn.axisgrid.FacetGrid at 0x1a23de3828>




![png](/assets/images/loan_prediction_feature_engineering/output_36_1.png)



```python
train['Annual Income * Term'] = train['Annual Income'] * train['Term']
```


```python
train['Current Loan Amount * Term'] = train['Current Loan Amount'] * train['Term']
```

**Min-Max scaling to avoid large numbers from negatively impact the models.** 


```python
# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
```


```python
# Scale feature
names = ['Current Loan Amount','Annual Income','Monthly Debt',
         'Current Credit Balance','Maximum Open Credit','Annual Income * Term',
         'Current Loan Amount * Term']

for name in names:
    train[name] = minmax_scale.fit_transform(train[[name]].astype(float))
```

**Adjusting null and inf values created by division.**


```python
train['Current Credit Usage'].fillna(1,inplace=True)
train['Current Credit Usage'] = train['Current Credit Usage'].apply(lambda x: 1 if np.isinf(x) else x)
```

## Preparing for Modeling

**Selecting numeric columns and dropping remaining null rows.**


```python
train_numbers = train.select_dtypes(['number'])
```


```python
train_numbers = train_numbers.dropna()
```

**Assigning features to X and target variable to y**


```python
X = train_numbers.drop(['Loan Status','Credit Score'],axis=1)
```


```python
y = train_numbers['Loan Status']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
```

**Using SMOTE to account for loan status class imbalance**


```python
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)
```

## Model Testing

**Random Forest had the highest F1 score but this could be due to overfitting of the model.**


```python
joeAssess(X_train, y_train, X_test, y_test, models)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RandomF</th>
      <td>0.833611</td>
      <td>0.893390</td>
      <td>0.913904</td>
      <td>0.903531</td>
    </tr>
    <tr>
      <th>GradientB</th>
      <td>0.804042</td>
      <td>0.921335</td>
      <td>0.842061</td>
      <td>0.879916</td>
    </tr>
    <tr>
      <th>DescisionT</th>
      <td>0.786174</td>
      <td>0.889340</td>
      <td>0.855680</td>
      <td>0.872185</td>
    </tr>
    <tr>
      <th>KNeighbors</th>
      <td>0.648904</td>
      <td>0.898905</td>
      <td>0.662743</td>
      <td>0.762967</td>
    </tr>
    <tr>
      <th>Bernoulli NB</th>
      <td>0.646024</td>
      <td>0.974232</td>
      <td>0.600718</td>
      <td>0.743184</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.641703</td>
      <td>0.978812</td>
      <td>0.592589</td>
      <td>0.738237</td>
    </tr>
  </tbody>
</table>
</div>



**Rescoring second best model using k-fold cross validation.**


```python
gbc = GradientBoostingClassifier(n_estimators=100)
```


```python
kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
```


```python
accuracy = []
precision = []
recall = []
f1 = []
names = []

for train_index, test_index in kf.split(train_numbers):

    X_train = train_numbers.iloc[train_index].drop(['Loan Status','Credit Score'],axis=1)
    y_train = train_numbers.iloc[train_index]['Loan Status']
    
    X_test = train_numbers.iloc[test_index].drop(['Loan Status','Credit Score'],axis=1)
    y_test = train_numbers.iloc[test_index]['Loan Status']
    
    gbc.fit(X_train, y_train)
    predicted_y = gbc.predict(X_test)

    first_score = accuracy_score(y_test, predicted_y)
    second_score = precision_score(y_test, predicted_y)
    third_score = recall_score(y_test, predicted_y)
    fourth_score = f1_score(y_test, predicted_y)

    accuracy.append(first_score)
    precision.append(second_score)
    recall.append(third_score)
    f1.append(fourth_score)
        
metrics = pd.DataFrame(columns = ['Accuracy','Precision','Recall','F1'])

metrics['Accuracy'] = accuracy
metrics['Precision'] = precision
metrics['Recall'] = recall
metrics['F1'] = f1

metrics.replace(1,0,inplace = True)
```


```python
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.859175</td>
      <td>0.869454</td>
      <td>0.983241</td>
      <td>0.922853</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.855478</td>
      <td>0.865218</td>
      <td>0.983684</td>
      <td>0.920655</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.860473</td>
      <td>0.868867</td>
      <td>0.985521</td>
      <td>0.923525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.853092</td>
      <td>0.862159</td>
      <td>0.984305</td>
      <td>0.919192</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.860834</td>
      <td>0.869622</td>
      <td>0.984777</td>
      <td>0.923624</td>
    </tr>
  </tbody>
</table>
</div>



**Gradient boosting classifier received at mean F1 of 0.922**


```python
metrics.F1.mean()
```




    0.9219698558136775




```python
gbc.fit(X,y)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  n_iter_no_change=None, presort='auto', random_state=None,
                  subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False)



**Showing top 10 important features of the gbc model**


```python
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = gbc.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features[-10:].plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1113a45c0>




![png](/assets/images/loan_prediction_feature_engineering/output_66_1.png)



```python
features.sort_values('importance',ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Was Deliquent</th>
      <td>0.687273</td>
    </tr>
    <tr>
      <th>Annual Income * Term</th>
      <td>0.117959</td>
    </tr>
    <tr>
      <th>Credit Score Ranges</th>
      <td>0.083006</td>
    </tr>
    <tr>
      <th>Monthly Debt/Income</th>
      <td>0.040944</td>
    </tr>
    <tr>
      <th>Current Loan Amount</th>
      <td>0.015069</td>
    </tr>
    <tr>
      <th>Annual Income</th>
      <td>0.012947</td>
    </tr>
    <tr>
      <th>Maximum Open Credit</th>
      <td>0.007188</td>
    </tr>
    <tr>
      <th>Months left</th>
      <td>0.006840</td>
    </tr>
    <tr>
      <th>Home Ownership_Rent</th>
      <td>0.006543</td>
    </tr>
    <tr>
      <th>Current Credit Usage</th>
      <td>0.005696</td>
    </tr>
  </tbody>
</table>
</div>



**Viewing correlation to compare with important features**


```python
train.corr()['Loan Status'].sort_values(ascending=False)
```




    Loan Status                     1.000000
    Credit Score                    0.250238
    Credit Score Ranges             0.214150
    Term                            0.150068
    Annual Income * Term            0.103351
    Current Loan Amount * Term      0.087155
    Current Credit Balance          0.033875
    Annual Income                   0.029791
    Purpose_Buy a Car               0.023394
    Maximum Open Credit             0.008227
    Purpose_Educational Expenses    0.005538
    Months left                     0.005524
    Purpose_Home Improvements       0.003241
    Purpose_Debt Consolidation      0.002213
    Purpose_Take a Trip             0.001801
    Purpose_Buy House               0.001708
    Bankruptcies                    0.001678
    Purpose_Other                   0.000060
    Purpose_Medical Bills          -0.004940
    Home Ownership_Own Home        -0.008086
    Years in current job           -0.010652
    Home Ownership_Rent            -0.022351
    Years of Credit History        -0.022552
    Number of Credit Problems      -0.023289
    Tax Liens                      -0.026371
    Monthly Debt                   -0.041890
    Number of Open Accounts        -0.050758
    Current Credit Usage           -0.053648
    Current Loan Amount            -0.065002
    Monthly Debt/Income            -0.104169
    Was Deliquent                  -0.385308
    Name: Loan Status, dtype: float64


