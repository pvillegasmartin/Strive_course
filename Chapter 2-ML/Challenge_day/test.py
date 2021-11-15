import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTENC
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Data enhancement based on gender with best features
def enhancement(data, porp):
    gen_data =data
    for sex in data['sex'].unique():
        gender_data = gen_data[gen_data['sex']==sex]

        thalachh_std = gender_data['thalachh'].std()/porp
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

data = pd.read_csv(r'C:\Users\Pablo\Desktop\STRIVE AI\Strive_course\Chapter 2\Challenge_day\heart.csv')

accuracies = {'m1':[],'m2':[],'m3':[],'m4':[]}

for porp in range(3,4):

    df = enhancement(data, porp)
    df = df.sample(frac=0.25,random_state=42)
    new_data = pd.concat([data, df])

    x_train, x_test, y_train, y_test = train_test_split(new_data.values[:,:-1], new_data.values[:,-1], test_size=0.3, random_state = 42, stratify=new_data.values[:,-1])



    cat_indx =[1,2,5,6,8,10,11,12]
    sm = SMOTENC(categorical_features= cat_indx, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    """
    X_train_res, y_train_res = x_train, y_train
    """
    print(len(data)/(len(X_train_res)+len(x_test)))




    #Its not mandatory in tree algorithms but it helps in computational timing
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train_res)
    x_test_scaled = scaler.transform(x_test)


    clf_1 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000,random_state=42)
    clf_2 = XGBClassifier(random_state=42)
    clf_3 = CatBoostClassifier(verbose=False,max_depth=4,random_state=42)

    params_4 = {'kernel':['rbf','linear','poly']}
    clf4 = svm.SVC(random_state=42)
    clf_4 = GridSearchCV(clf4,params_4)

    #params = {'max_depth':[3,4,5,6,7,8]}
    #clf_3 = GridSearchCV(clf3, params)

    clf_1.fit(x_train_scaled,y_train_res)
    clf_2.fit(x_train_scaled,y_train_res)
    clf_3.fit(x_train_scaled,y_train_res)
    clf_4.fit(x_train_scaled, y_train_res)

    preds_1 = clf_1.predict(x_test_scaled)
    preds_2 = clf_2.predict(x_test_scaled)
    preds_3 = clf_3.predict(x_test_scaled)
    preds_4 = clf_4.predict(x_test_scaled)
    print(f'---------PROP={porp}-------------')
    print(accuracy_score(y_test,preds_1))
    print(confusion_matrix(y_test, preds_1))
    print(accuracy_score(y_test,preds_2))
    print(confusion_matrix(y_test, preds_2))
    print(accuracy_score(y_test,preds_3))
    print(confusion_matrix(y_test, preds_3))
    #print(clf_3.best_params_)
    print(accuracy_score(y_test, preds_4))
    print(confusion_matrix(y_test, preds_4))
    print(clf_4.best_params_)

    accuracies['m1'].append(accuracy_score(y_test,preds_1))
    accuracies['m2'].append(accuracy_score(y_test, preds_2))
    accuracies['m3'].append(accuracy_score(y_test, preds_3))
    accuracies['m4'].append(accuracy_score(y_test, preds_4))

acc_porp={}
for i in range(3,3+len(accuracies["m1"])):
    acc_porp[i]= (accuracies["m1"][i-3]+accuracies["m2"][i-3]+accuracies["m3"][i-3]+accuracies["m4"][i-3])/4

print(acc_porp)

print(f'Model 1: {np.array(accuracies["m1"]).mean()}, Model 2: {np.array(accuracies["m2"]).mean()}, Model 3: {np.array(accuracies["m3"]).mean()}, Model 4: {np.array(accuracies["m4"]).mean()}')
