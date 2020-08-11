import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("https://raw.githubusercontent.com/"\
                "PacktWorkshops/"\
                "The-Data-Analysis-Workshop/master/"\
                "Chapter05/Datasets/online_shoppers_intention.csv")
print(df.head(10))
print(df.info())
print(df.iloc[:, [3,6]].values)
print(df.isnull().sum())

plt.figure(figsize =(6,6))
sns.countplot(df['Revenue'])
plt.title('Baseline Revenue Conversion', fontsize =20)
plt.show()

print(df['Revenue'].value_counts())

print(df['Revenue'].value_counts(normalize = True))

plt.figure(figsize =(6,6))
sns.countplot(df['VisitorType'])
plt.title('Visitor Type wise Distribution', fontsize = 20)
plt.show()

#calculation exact number of each visitor type
print(df['VisitorType'].value_counts())
print()
print(df['VisitorType'].value_counts(normalize=True))

plt.figure(figsize =(6,6))
sns.countplot(df['TrafficType'])
plt.title('Traffic Type wise Distribution', fontsize = 20)
plt.show()

print(df['TrafficType'].value_counts(normalize=True))

plt.figure(figsize =(6,6))
sns.countplot(df['Weekend'])
plt.title('Weekend Session distribution', fontsize = 20)
plt.show()

print(df['Weekend'].value_counts())
print(df['Weekend'].value_counts(normalize = True))

plt.figure(figsize=(6,6))
sns.countplot(df['Region'])
plt.title('Region wise distribution', fontsize = 20)
plt.show()

print(df['Region'].value_counts())
print(df['Region'].value_counts(normalize = True))

plt.figure(figsize=(6,6))
sns.countplot(df['Browser'])
plt.title('Browser wise session distribution', fontsize = 20)
plt.show()

print(df['Browser'].value_counts())
print(df['Browser'].value_counts(normalize = True))

plt.figure(figsize =(6,6))
sns.countplot(df['OperatingSystems'])
plt.title('OS wise session distribution', fontsize = 20)
plt.show()

print(df['OperatingSystems'].value_counts())
print(df['OperatingSystems'].value_counts(normalize = True))

plt.figure(figsize =(6,6))
sns.countplot(df['Administrative'])
plt.title('Administrative Pageview Distribution', fontsize = 16)
plt.show()

plt.figure(figsize =(6,6))
sns.countplot(df['Informational'])
plt.title('Information Pageview Distribution', fontsize = 16)
plt.show()

plt.figure(figsize =(6,6))
sns.countplot(df['SpecialDay']) 
plt.title('Special Day session Distribution', fontsize = 16)
plt.show()

print(df['SpecialDay'].value_counts(normalize=True))

#bivariate analysis
plt.figure(figsize =(10,8))
sns.catplot(x= 'Revenue', col ='VisitorType', data = df,col_wrap =3, kind ='count', height = 5, aspect=1)
plt.show()

plt.figure(figsize = (10, 8))
sns.countplot(x='TrafficType', hue ='Revenue', data= df)
plt.show()

plt.figure(figsize =(10, 8))
sns.countplot(x="Region", hue="Revenue", data=df)
plt.show()

plt.figure(figsize =(10,8))
sns.countplot(x="Browser", hue="Revenue", data=df)
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x="OperatingSystems", hue="Revenue", data=df)
plt.show()

plt.figure(figsize =(10,8))
sns.countplot(x="Month", hue="Revenue",data=df,order=['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.show()

plt.figure(figsize =(10,10))
sns.set(style='whitegrid')
sns.lmplot(x='BounceRates', y= 'ExitRates', data=df)
plt.show()

plt.figure(figsize = (10, 10))
sns.set(style ='whitegrid')
sns.lmplot(x='PageValues', y= 'BounceRates', data= df)
plt.show()

plt.figure(figsize = (10, 10))
sns.set(style ='whitegrid')
sns.lmplot(x='PageValues', y= 'ExitRates', data= df)
plt.show()

plt.figure(figsize = (10, 10))
sns.set(style ='whitegrid')
sns.lmplot(x='Administrative', y= 'Administrative_Duration',hue ='Revenue' ,data= df)
plt.show()


plt.figure(figsize = (10, 10))
sns.set(style ='whitegrid')
sns.lmplot(x='Informational', y= 'Informational_Duration',hue ='Revenue' ,data= df)
plt.show()

x= df.iloc[:, [3,6]].values
wcss =[]

for i in range(1, 11):
    km = KMeans(n_clusters =i,init ='k-means++', max_iter=300, n_init =10,algorithm ='elkan',random_state=0, tol = 0.001)
    km.fit(x)
    km.labels_
    wcss.append(km.inertia_)

plt.rcParams['figure.figsize'] = (15,7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The elbow method', fontsize = 20)
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

km = KMeans(n_clusters=2,init ='k-means++', max_iter = 300, algorithm='elkan', random_state=0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means ==0, 0], x[y_means ==0, 1],s=100, c='pink', label = 'Un-interested customers')
plt.scatter(x[y_means ==1, 0], x[y_means ==1, 1], s=100, c='yellow', label ='target customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=50, c='blue', label ='centroids')
plt.title('Informational Duration vs Exit Rates', fontsize = 20)
plt.grid()
plt.xlabel('Informational Duration')
plt.ylabel('Exit Rates')
plt.legend()
plt.show()


