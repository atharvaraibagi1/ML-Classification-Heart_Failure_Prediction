
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
data.head()
data.shape
data.age.describe()

from scipy.stats import norm
plt.hist(data.age, bins = 15, rwidth=0.5, density = True);
rng = np.arange(0,110)
plt.plot(rng, norm.pdf(rng, data.age.mean(),data.age.std()));
upper_age_limit= data.age.quantile(0.999)
upper_age_limit
data2 = data[(data.age<upper_age_limit)]
data.shape[0]- data2.shape[0]
plt.hist(data2.age, bins = 15, rwidth=0.5, density = True);
rng = np.arange(0,110)
plt.plot(rng, norm.pdf(rng, data2.age.mean(),data2.age.std()));
data.anaemia.unique()
data.creatinine_phosphokinase.describe()
plt.hist(data2.creatinine_phosphokinase, bins = 15, rwidth=0.5, density = True);
rng = np.arange(-4000,data2.creatinine_phosphokinase.max())
plt.plot(rng, norm.pdf(rng, data2.creatinine_phosphokinase.mean(),data2.creatinine_phosphokinase.std()));
upper_creatinine_phosphokinase_limit = data2.creatinine_phosphokinase.quantile(0.98)
upper_creatinine_phosphokinase_limit
data3 = data2[(data2.creatinine_phosphokinase<upper_creatinine_phosphokinase_limit)]
data2.shape[0] - data3.shape[0]
plt.hist(data3.creatinine_phosphokinase, bins = 15, rwidth=0.5, density = True);
rng = np.arange(-4000,data3.creatinine_phosphokinase.max())
plt.plot(rng, norm.pdf(rng, data3.creatinine_phosphokinase.mean(),data3.creatinine_phosphokinase.std()));
data3.shape
data3.head()
data3.diabetes.unique()
data3.ejection_fraction.unique()
plt.hist(data3.ejection_fraction, bins = 15, rwidth=0.5, density = True);
rng = np.arange(0,data3.ejection_fraction.max())
plt.plot(rng, norm.pdf(rng, data3.ejection_fraction.mean(),data3.ejection_fraction.std()));
data3[data3.ejection_fraction>65]
upper_ejection_fraction_limit = data3.ejection_fraction.quantile(0.9931)
upper_ejection_fraction_limit
data4 = data3[(data3.ejection_fraction<upper_ejection_fraction_limit)]
data3.shape[0] - data4.shape[0]
plt.hist(data4.ejection_fraction, bins = 15, rwidth=0.5, density = True);
rng = np.arange(0,data4.ejection_fraction.max())
plt.plot(rng, norm.pdf(rng, data4.ejection_fraction.mean(),data4.ejection_fraction.std()));
data4.shape
data4.head()
data4.high_blood_pressure.unique()
data4.platelets.nunique()
plt.hist(data4.platelets, bins = 15, rwidth=0.5, density = True);
rng = np.arange(0,data4.platelets.max())
plt.plot(rng, norm.pdf(rng, data4.platelets.mean(),data4.platelets.std()));
upper_platelets_limit = data4.platelets.quantile(0.993)
upper_platelets_limit
data5 = data4[(data4.platelets<upper_platelets_limit)]
data5.shape
plt.hist(data5.platelets, bins = 15, rwidth=0.5, density = True);
rng = np.arange(0,data5.platelets.max())
plt.plot(rng, norm.pdf(rng, data5.platelets.mean(),data5.platelets.std()));
data4.shape[0] -data5.shape[0]
data5.head()
data5.serum_creatinine.describe()
plt.hist(data5.serum_creatinine, bins = 15, rwidth=0.5, density = True);
rng = np.arange(-5,data5.serum_creatinine.max())
plt.plot(rng, norm.pdf(rng, data5.serum_creatinine.mean(),data5.serum_creatinine.std()));
upper_serum_creatinine_limit = data.serum_creatinine.quantile(0.995)
upper_serum_creatinine_limit
data6 = data5[(data5.serum_creatinine<upper_serum_creatinine_limit)]
data5.shape[0] - data6.shape[0]
plt.hist(data6.serum_creatinine, bins = 15, rwidth=0.5, density = True);
rng = np.arange(-5,data6.serum_creatinine.max())
plt.plot(rng, norm.pdf(rng, data6.serum_creatinine.mean(),data6.serum_creatinine.std()));
data6.head()
data6.serum_sodium.unique()
data6.serum_sodium.describe()
plt.hist(data6.serum_sodium, bins = 15, rwidth=0.5, density = True);
rng = np.arange(data6.serum_sodium.min(),155)
plt.plot(rng, norm.pdf(rng, data6.serum_sodium.mean(),data6.serum_sodium.std()));
data6.head(2)
data6.sex.unique()
data6.smoking.unique()
data6.time.unique()
data6.time.describe()
data6.DEATH_EVENT.unique()
X = data6.drop(["DEATH_EVENT"],axis =1)
X.head()
y = data6["DEATH_EVENT"]
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        },
        'Random_forest_classifier':{
            'model': RandomForestClassifier(),
            'params':{
                'criterion' : ['gini','entropy']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)
model = RandomForestClassifier(criterion='entropy')
model.fit(X_train,y_train)
model.score(X_test,y_test)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
score = cross_val_score(RandomForestClassifier(criterion='entropy'),X,y,cv =cv)
score.mean()
model
y_pred = model.predict(X_test)
y_pred
import pickle
with open("heart.pickle","wb") as f:
    pickle.dump(model,f)
