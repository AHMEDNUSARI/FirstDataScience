import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy
from sklearn import linear_model
from sklearn import neighbors



#load data
data = pandas.read_csv("insurance.csv")
data.head(5)
# reindexing data columns
data=data[['age','sex','bmi','children','region','charges','smoker']]
#Encoding string elemnts to numbers
categ = [ 'sex','region','smoker']
le = LabelEncoder()
data[categ] = data[categ].apply(le.fit_transform)
data.head(5)
data=data.apply(numpy.ceil)
data=data.to_numpy()

# creating data
X, y = data[:, :6], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)



# Decision Trees classifier 
classifier_DecisionTree = DecisionTreeClassifier(**{'random_state': 0, 'max_depth': 4})
classifier_DecisionTree.fit(X_train, y_train)
y_test_pred = classifier_DecisionTree.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier_DecisionTree performance on training dataset\n")
print(classification_report(y_train, classifier_DecisionTree.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier_DecisionTree performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()

# plot confusion matrix
plot_confusion_matrix(classifier_DecisionTree, X_test, y_test)  
plt.title('Confusion matrix Decision Tree Classifier')
plt.show()

# Rastgele Orman (Random Forest)

classifier_RandomForest = RandomForestClassifier(n_jobs=2, random_state=0)
classifier_RandomForest.fit(X_train, y_train)

y_test_pred = classifier_RandomForest.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier_RandomForest performance on training dataset\n")
print(classification_report(y_train, classifier_RandomForest.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier_RandomForest performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()

# plot confusion matrix
plot_confusion_matrix(classifier_RandomForest, X_test, y_test)  
plt.title('Confusion matrix Random Forest Classifier')
plt.show()

"Lojistik Regresyon"

classifier_logisticREG = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier_logisticREG.fit(X_train, y_train)
y_test_pred = classifier_logisticREG.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier_logisticREG performance on training dataset\n")
print(classification_report(y_train, classifier_logisticREG.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier_logisticREG performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()

# plot confusion matrix
plot_confusion_matrix(classifier_logisticREG, X_test, y_test)  
plt.title('Confusion matrix logistic REG Classifier')
plt.show()


# K-en yakın komşu (k-Nearest Neighbors classifier)

classifier_KNeighborsClassifier = neighbors.KNeighborsClassifier(12, weights='distance')

# Train the K Nearest Neighbors model
classifier_KNeighborsClassifier.fit(X_train, y_train)

y_test_pred = classifier_KNeighborsClassifier.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier_KNeighborsClassifier performance on training dataset\n")
print(classification_report(y_train, classifier_KNeighborsClassifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier_KNeighborsClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()

# plot confusion matrix

plot_confusion_matrix(classifier_KNeighborsClassifier, X_test, y_test)  
plt.title('Confusion matrix K Neighbors Classifier')
plt.show()
