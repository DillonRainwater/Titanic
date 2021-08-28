#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

train_data = pd.read_csv(r"C:\Users\Dillon Rainwater\Documents\Python\Kaggle Competitions\Titanic\titanic data\train.csv")
test_data = pd.read_csv(r"C:\Users\Dillon Rainwater\Documents\Python\Kaggle Competitions\Titanic\titanic data\test.csv")

#%%
train_data.head()

#%%
num_rows = len(train_data)
num_rows

#%%
#finding missing values
missing_data_train = train_data.isna().sum()
missing_data_train = missing_data_train[missing_data_train > 0]
missing_data_train

#%%
missing_data_test = test_data.isna().sum()
missing_data_test = missing_data_test[missing_data_test > 0]
missing_data_test

#%%
survived_data = train_data.loc[train_data['Survived'] == 1]
survived_data.mean(skipna=True)

#%%
died_data = train_data.loc[train_data['Survived'] == 0]
died_data.mean(skipna=True)

#%%
sb.barplot(x="Sex", y="Survived", data=train_data)

#%%
sb.barplot(x='Survived', y='Fare', data=train_data)

#%%
sb.barplot(x='Pclass', y='Survived', data=train_data)

#%%
sb.barplot(x='Pclass', y='Fare', data=train_data)

#%%
sb.barplot(x='Embarked', y='Survived', data=train_data)

#%%
sb.barplot(x="Embarked", y='Fare', data=train_data)

#%%
sb.barplot(x='Survived', y='Fare', hue='Sex', data=train_data)

#%%
sb.barplot(x='Pclass', y='Fare', hue='Survived', data=train_data)

#%%
g = sb.FacetGrid(data=train_data, row="Pclass", col="Sex", margin_titles=True)
g.map(sb.barplot, "Survived", "Fare")

#%%
for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sb.catplot(x = col, y = 'Survived', data = train_data, kind = 'point')

#%%
g = sb.FacetGrid(data=train_data, col="Survived", margin_titles=True)
g.map(sb.histplot, "Age")

#%%
sb.heatmap(train_data.corr(), annot = True)
plt.title('Correlation between Features in train_data')

#%%
# Dealing with missing values
train_data['cabin_missing'] = np.where(train_data['Cabin'].isnull(), 1, 0)
sb.barplot(x='cabin_missing', y='Survived', data=train_data)
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
train_data.drop(train_data.loc[train_data['Embarked'].isna() == True].index, inplace=True)

# apply to test_data as well
test_data['cabin_missing'] = np.where(test_data['Cabin'].isnull(), 1, 0)
test_data['Age'].fillna(test_data['Age'].mean(), inplace = True)

test_data.drop(test_data.loc[test_data['Fare'].isna() == True].index, inplace=True)

# %%
y = train_data["Survived"]

features = ["Pclass", "Sex", "Fare", "Embarked", "cabin_missing", "Age", 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test_data = pd.get_dummies(test_data[features])

X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state = 0)

# normalizing data
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

#%%
def get_RandomForest_scores(n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=4)
    scores = cross_val_score(model, X, y, cv=5)
    return(scores)

for n in [5,10,50,100,250,500]:
    print(get_RandomForest_scores(n))
    print(get_RandomForest_scores(n).mean())

#%%
def get_GradientBossting_scores(n_estimators):
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
    scores = cross_val_score(model, X, y, cv=5)
    return(scores)

for n in [5,10,50,100,250,500]:
    print(get_GradientBossting_scores(n))
    print(get_GradientBossting_scores(n).mean())


#%%
# Using GridSearchCV to find best hyperparameters

rf = RandomForestClassifier(n_jobs = -1, random_state = 0)
params = {
    'n_estimators': [5, 50, 100, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, params, cv = 5, n_jobs = -1)
cv.fit(X_train_norm, Y_train)
cv.best_params_

#%%

gb = GradientBoostingClassifier(random_state = 0)
params = {
    'n_estimators': [5, 50, 100, 250],
    'max_depth': [2, 4, 8, 16, 32, None],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}

cv = GridSearchCV(gb, params, cv = 5, n_jobs = -1)
cv.fit(X_train_norm, Y_train)
cv.best_params_

#%%
model = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=0)
model.fit(X_train_norm, Y_train)
score = model.score(X_train_norm, Y_train)
print(score)
rf_predictions = model.predict(X_test_norm)

#%%
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, n_estimators=50)
model.fit(X_train_norm, Y_train)
score = model.score(X_train_norm, Y_train)
print(score)
gb_predictions = model.predict(X_test_norm)

#%%
cm = confusion_matrix(Y_test, rf_predictions)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()

#%%
cm = confusion_matrix(Y_test, gb_predictions)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()

#%%
#predictions output to submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': rf_predictions})
output.to_csv('titanic_submission.csv', index=False)
