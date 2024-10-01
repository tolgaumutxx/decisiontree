from sklearn.tree import DecisionTreeClassifier
import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

flags = pd.read_csv("flags.csv",header = 0)

labels = flags['Landmass']
data = flags[['Red','Green','Blue','Gold','White','Black','Orange',"Circles",'Crosses','Saltires','Quarters','Sunstars','Crescent','Triangle']]
print(flags.columns)

X_train,X_test,y_train,y_test = train_test_split(data,labels,random_state = 1, test_size = 0.25)
scores = []
for i in range(1,21):
  tree = DecisionTreeClassifier(random_state = 1,max_depth = i)
  tree.fit(X_train,y_train)

  scores.append(tree.score(X_test,y_test))

plt.plot(range(1,21),scores)
plt.show()

"""
To find the index with the most gain if chosen
def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    # Create features here

    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_gain, best_feature

"""

#Gini index
#def gini(dataset):
#  impurity = 1
#  label_counts = Counter(dataset)
#  for label in label_counts:
#    prob_of_label = label_counts[label] / len(dataset)
#    impurity -= prob_of_label ** 2
#  return impurity


classifier = DecisionTreeClassifier()

#decision trees are greedy, making an optmal tree is hard