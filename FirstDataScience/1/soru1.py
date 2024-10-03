""" soru 1"""

""" Veri Ön işleme (Çift değerlerin silinmesi, Alakasız değerlerin silinmesi, Tutarsız
                   değerlerin kaldırılması, İstenmeyen sütunun veya satırın kaldırılması, İstenmeyen
                   sütunun veya satırın kaldırılması, Eksik değerlerin silinmesi, Aykırı değerlerin
                   kaldırılması)"""

import pandas
import numpy



"Çift değerlerin silinmesi"

Data_set=pandas.DataFrame(([3,5,7,2,7,9,0,3,5,8,4,4]))
Data_set.drop_duplicates()


" İstenmeyen sütunun kaldırılması"

data=pandas.read_csv("company.csv")
data.head(5)
data1=data.drop(['TV','Radio'],axis=1)

" Eksik değerlerin silinmesi"

data=pandas.read_csv("ADANIPORTS.csv")
data.isnull().sum()
data2=data.dropna(axis='columns')
data2.isnull().sum()

" Aykırı değerlerin kaldırılması)"

data=pandas.read_csv("company.csv")
data.head(10)
data.loc[data['TV']>50]
"========================================================================================================================"

"  ExtraTreesClassifier ile Öznitelik önemi (Feature importance) çıkarınız "
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import utils
import seaborn
import pandas
import numpy



#data = pandas.read_csv("train.csv")
#data=pandas.DataFrame(numpy.random.randint(20, size=(100, 6)))
data = pandas.read_csv("company.csv")
data=data.apply(numpy.ceil)
X = data.iloc[:,0:6]
y = data.iloc[:,-1]
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class
# feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pandas.Series(model.feature_importances_, index=X.
columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

"======================================================================"
"Veri Setindeki en iyi öznitelikleri skorlandırınız, çıkarınız (SelectKBest)"

import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pandas.read_csv("company.csv")
X = data.iloc[:,0:20] #independent columns
y = data.iloc[:,-1] #pick last column for the target feature
#apply SelectKBest class to extract top 5 best features
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pandas.DataFrame(fit.scores_)
dfcolumns = pandas.DataFrame(X.columns)
scores = pandas.concat([dfcolumns,dfscores],axis=1)
scores.columns = ['specs','score']
print(scores.nlargest(4,'score')) #print the 5 best features

"==========================================================================="
"Korelasyon ısı haritası çıkarınız (Correlation heat map)"
import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import utils
import seaborn

data = pandas.read_csv("company.csv")
correlation_matrix = data.corr()
top_corr_features = correlation_matrix.index
plt.figure(figsize=(4,4))
#plot heat map
g=seaborn.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
"==========================================================================="
" Normal dağılıma sahip olmayan verileri standartlaştırınız."
import pandas
import numpy
data = pandas.read_csv("company.csv")                       
                       
                       #Standartlaştırma                       
data2 = (data- data.mean()) / data.std()
print(data2)
                     
                       #Normalize etme
data3 = (data - data.min()) / (data.max() - data.min())
print(data3)

"==========================================================================="

"Veri üzerinde temel istatistik bilgileri çıkarınız"
import pandas
import numpy
data = pandas.read_csv("company.csv")
print(data.info())
print(data.shape)
 

#mod
data.mode()

#medyan
data.median()  
      
#aritmetik ortalama
data.mean()

#standart sapma
data.std()

#varyans
data.var()

#kovaryans
data.cov()

#korelasyon
data.corr()
        




