#ECET392_S22_Mar23_1.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn import cross_validation
#from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier



#We will use data in the data_decision_trees.txt . The first two values correspond to the input data and the last value corresponds to the target label.
#Let's load the data from that file:
# Load input data
input_file = 'data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
np.shape(X), np.shape(y)
#((360, 2), (360,))

# Separate input data into two classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Visualize input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')
plt.show()

# Split data into training and testing datasets 
#X_train,X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Decision Trees classifier
params = {'random_state': 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
#visualize_classifier(classifier, X_train, y_train, 'Training dataset')
visualize_classifier(classifier, X_train, y_train)
y_test_pred = classifier.predict(X_test)
#visualize_classifier(classifier, X_test, y_test, 'Test dataset')
visualize_classifier(classifier, X_test, y_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")


