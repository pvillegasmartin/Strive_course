import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, accuracy_score

# Load dataset
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','target']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None, index_col=False, names=attributes)

# Preparing the data
df = df.replace(['vhigh', 'high', 'med', 'low', 'small', 'big'], [4, 3, 2, 1, 1, 3])
X, y = df.iloc[:,:-1], df.target
X.persons = X.persons.replace('more',5)
X.doors = X.doors.replace('5more',5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Model Selection
'''
# STEP 1: which parameters values
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[2, 10000]}
classifier = DecisionTreeClassifier()
clf = GridSearchCV(classifier, parameters)
clf.fit(X_train, y_train)
best_parameters = clf.best_params_ # we see best parameters are criterion='gini' and max_depth=None
print(best_parameters)

# STEP 2: verify is a good model with cross validation
classifier = DecisionTreeClassifier(criterion='gini', max_depth=None)
cv = cross_validate(classifier, X_train, y_train, cv=5)
print(cv['test_score'].mean()) # accuracy mean of 0.9751928946195261

'''
clf = DecisionTreeClassifier(criterion='gini', max_depth=None)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Plots
#plt.figure(figsize=(10, 10))
#plot_tree(clf, filled=True, rounded=True, class_names=["unacc", "acc", "good", "vgood"], feature_names=X.columns)
plot_confusion_matrix(clf, X_test, y_test, display_labels=["unacc", "acc", "good", "vgood"])
plt.show()
cv_result_train = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1, scoring="accuracy")
cv_result_test = cross_val_score(clf, X_test, y_test, cv=5, n_jobs=-1, scoring="accuracy")
acc_score = accuracy_score(y_test, predictions)
print(cv_result_train)
print(cv_result_test)
