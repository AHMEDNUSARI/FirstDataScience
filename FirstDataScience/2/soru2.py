"Aşağıdaki veri üzerinde sınıflandırıcılar ile tahminleme işlemini yapınız"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import numpy 
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier 
from sklearn.preprocessing import StandardScaler
import neurolab 


data=pandas.read_csv("bank-full.csv",sep=';')
data.head(5)

#display columns
#data.columns
data.isnull().sum()

categ = ['job','marital','education','default','balance','housing','loan','contact','month','poutcome','y']
le = LabelEncoder()
data[categ] = data[categ].apply(le.fit_transform)
data.head(10)
#data Spiltting 
data=data.to_numpy()
X, y = data[:, :16], data[:, 16:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)



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

#######################################################################

"Naïve Bayes"
classifier_NaiveBayed = GaussianNB()

# Train the classifier
classifier_NaiveBayed.fit(X_train, y_train)

# Predict the values for training data
y_test_pred = classifier_NaiveBayed.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier Naive Bayed performance on training dataset\n")
print(classification_report(y_train, classifier_NaiveBayed.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier Naive Bayed performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")


# plot confusion matrix
plot_confusion_matrix(classifier_NaiveBayed, X_test, y_test)  
plt.title('Confusion matrix classifier Naive Bayed')
plt.show()
#################################################################################


"Destek Vektör Makineleri (Support Vector Machines-SVM)"

# Create SVM classifier
classifier_SVM = OneVsOneClassifier(LinearSVC(random_state=0))

# Train the classifier 
classifier_SVM.fit(X_train, y_train)

y_test_pred = classifier_SVM.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier SVM performance on training dataset\n")
print(classification_report(y_train, classifier_SVM.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier SVM performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()

# plot confusion matrix
plot_confusion_matrix(classifier_SVM, X_test, y_test)  
plt.title('Confusion matrix classifier SVM')
plt.show()

##########################################################################
#kNN

from sklearn import neighbors

# Create a K Nearest Neighbors classifier model
KNeighbors_Classifier = neighbors.KNeighborsClassifier(10, weights='distance')

# Train the K Nearest Neighbors model
KNeighbors_Classifier.fit(X_train, y_train)

y_test_pred = KNeighbors_Classifier.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nclassifier K Neighbors performance on training dataset\n")
print(classification_report(y_train, KNeighbors_Classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nclassifier K Neighbors performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()

# plot confusion matrix
plot_confusion_matrix(KNeighbors_Classifier, X_test, y_test)  
plt.title('Confusion matrix classifier K Neighbors')
plt.show()

##############################################################################
