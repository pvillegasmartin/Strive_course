import os
from PIL import Image
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection, metrics
import time

def rotate_img (img):
    rotated = img.rotate(45)
    return rotated

def train_csv (path, rotate=False):
    df = pd.DataFrame()
    for folder in os.listdir(path):
        folder = os.path.join(path, folder)
        print(folder)
        for filename in os.listdir(folder):
            img = Image.open(os.path.join(folder, filename))
            img_array = np.array(img, dtype=float)
            img_array = img_array.reshape(784, )
            df_append = pd.DataFrame({'img': img_array}).T
            df_append['class'] = folder
            df = df.append(df_append)
            if rotate:
                img = rotate_img(img)
                img_array = np.array(img, dtype=float)
                img_array = img_array.reshape(784, )
                df_append = pd.DataFrame({'img': img_array}).T
                df_append['class'] = folder
                df = df.append(df_append)
    df.to_csv('numbers_train.csv', index=False, header=False)

def load_images_from_csv(file):
    numbers = pd.read_csv(file, header=None)
    x = numbers.iloc[:, :-1]
    y = numbers.iloc[:, -1]
    return x, y

if __name__=='__main__':
    """
    path = r'.\trainingSet\trainingSet'
    train_csv(path, rotate=False)
    """
    x, y = load_images_from_csv('numbers_train.csv')
    tree_classifiers = {
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100)
    }
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Bal Acc.', 'Time'])
    skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for model_name, model in tree_classifiers.items():
        start_time = time.time()

        # TRAIN AND GET PREDICTIONS USING cross_val_predict() and x,y
        pred = model_selection.cross_val_predict(model, x, y, cv=skf)
        total_time = time.time() - start_time

        results = results.append({"Model": model_name,
                                  "Accuracy": metrics.accuracy_score(y, pred) * 100,
                                  "Bal Acc.": metrics.balanced_accuracy_score(y, pred) * 100,
                                  "Time": total_time},
                                 ignore_index=True)

    print(results)
