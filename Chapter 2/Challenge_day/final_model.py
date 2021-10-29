
""""
Program to estimate whether the person has heart attack or not

"""

# importing required libraries

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTENC
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle


# Data enhancement based on gender with best features
def enhancement(data, porp):
    gen_data =data
    for sex in data['sex'].unique():
        gender_data = gen_data[gen_data['sex']==sex]

        thalachh_std = gender_data['thalachh'].std()/porp #Dividing standard-Deviation with a constant
        oldpeak_std = gender_data['oldpeak'].std()/porp
        caa_std = gender_data['caa'].std()/porp
        cp_std =gender_data['cp'].std()/porp
        j=0
        for i in gen_data[gen_data['sex']==sex].index:
            if j == 0:
                gen_data['thalachh'].values[i] += thalachh_std
                j==1
            else:
                gen_data['thalachh'].values[i] -= thalachh_std

            if j == 0:
                gen_data['oldpeak'].values[i] += oldpeak_std
                j == 1
            else:
                gen_data['oldpeak'].values[i] -= oldpeak_std

            if j == 0:
                gen_data['caa'].values[i] += caa_std
                j == 1
            else:
                gen_data['caa'].values[i] -= caa_std

            if j == 0:
                gen_data['cp'].values[i] += cp_std
                j == 1
            else:
                gen_data['cp'].values[i] -= cp_std

    return gen_data

# reading the data
data = pd.read_csv('heart.csv')


df = enhancement(data, 3) #enhanced dataframe
df = df.sample(frac=0.25,random_state=42) # sample 0.25% data from enhanced dataframe 
new_data = pd.concat([data, df]) # Adding enhanced data to orginal data

# splits
x_train, x_test, y_train, y_test = train_test_split(new_data.values[:,:-1], new_data.values[:,-1], test_size=0.3, random_state = 42, stratify=new_data.values[:,-1])

# Balancing
cat_indx =[1,2,5,6,8,10,11,12]
sm = SMOTENC(categorical_features= cat_indx, random_state=42)
X_train_res, y_train_res = sm.fit_resample(x_train, y_train)



#Its not mandatory in tree algorithms but it helps in computational timing
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train_res)
x_test_scaled = scaler.transform(x_test)


clf_2 = XGBClassifier(random_state=42) # XGB classifier model


clf_2.fit(x_train_scaled,y_train_res) #fit model

preds_2 = clf_2.predict(x_test_scaled) # predictions


#print(accuracy_score(y_test,preds_2)) # accuracy
#print(confusion_matrix(y_test, preds_2)) # confusion matrix

# updating the data columns with aa new column 
df1 = new_data.copy()
df1["new_column"] = df1['oldpeak']*df1['slp']

#print(df1.head())
X = df1.drop(["output"],axis = 1)
Y = df1.output
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_3= XGBClassifier(random_state=42)


clf_3.fit(X_train_scaled,Y_train)

preds_3 = clf_3.predict(X_test_scaled)
#print(accuracy_score(Y_test,preds_3))
#print(confusion_matrix(Y_test, preds_3))

filename = 'finalized_model.sav'
pickle.dump(clf_3, open(filename, 'wb'))

print(f'{len(data)*100/(len(X_train_res)+len(x_test)):.2f} % of final data is original')
print(f'We create the column oldpeak*slp because we see them are related. Accuracy pass from {accuracy_score(y_test,preds_2)*100:.2f}% to {accuracy_score(Y_test,preds_3)*100:.2f}%')