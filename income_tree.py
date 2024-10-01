import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


income_data = pd.read_csv('income.csv',header=0,delimiter=', ')
print(income_data.columns)

income_labels = income_data['income']
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data['country'] = income_data['native-country'].apply(lambda row: 1 if row == 'United-States' else 0)
#data = income_data[['age','education','capital-gain','hours-per-week','sex','race']]
data = income_data[['age','capital-gain','capital-loss','hours-per-week','sex-int','country']]
classifier = RandomForestClassifier(n_estimators=1000,random_state=5)

X_train,X_test,y_train,y_test = train_test_split(data,income_labels,test_size=0.2,random_state=1)

classifier.fit(X_train,y_train)
print(classifier.feature_importances_)

print(classifier.score(X_test,y_test))
est = classifier.estimators_[0]
plt.figure(figsize=(10, 8))  # Set the figure size for better visibility
plot_tree(est, filled=True, feature_names=['age','capital-gain','capital-loss','hours-per-week','sex-int','country'], class_names=['<50', '>50'])
plt.show()

#print(income_data['native-country'].value_counts())




