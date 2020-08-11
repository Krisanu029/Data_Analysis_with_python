import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel('default_credit.xls')
#print(df.head(10))
print(df.info())

print(df.describe().T)
print(df.isnull().sum())
print('SEX' + str(sorted(df['SEX'].unique())))
print('EDUCATION' + str(sorted(df['EDUCATION'].unique())))
print('MARRIAGE' + str(sorted(df['MARRIAGE'].unique())))
print('PAY_0 ' + str(sorted(df['PAY_0'].unique())))
print('default.payment.next.month ' + str(sorted(df['default payment next month'].unique())))
fill = (df.EDUCATION ==0) | (df.EDUCATION ==5) | (df.EDUCATION==6)
df.loc[fill, 'EDUCATION'] = 4
print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))
fill = (df.MARRIAGE == 0)
df.loc[fill, 'MARRIAGE'] = 2
print('MARRIAGE ' + str(sorted(df['MARRIAGE'].unique())))
df= df.rename(columns = {'default payment next month': 'DEFAULT',
                        'PAY_0': 'PAY_1'})
print(df.head())
sns.countplot(x="DEFAULT", data=df)
print(df['DEFAULT'].value_counts())
sns.countplot(x="SEX", data=df)
print(df['SEX'].value_counts())
sns.countplot(x="EDUCATION", data=df)
print(df['EDUCATION'].value_counts())
sns.countplot(x="MARRIAGE", data=df)
print(df['MARRIAGE'].value_counts())
sns.set(rc = {'figure.figsize':(15,10)})
edu = sns.countplot(x='SEX', hue ='DEFAULT', data= df)
edu.set_xticklabels(['Male', 'Female'])
plt.show()
k= pd.crosstab(df.SEX, df.DEFAULT, normalize='index',margins= True)
print(k)
sns.set(rc={'figure.figsize':(15,10)})
edu = sns.countplot(x='EDUCATION', hue='DEFAULT', data=df)
edu.set_xticklabels(['Graduate School','University',
                         'High School','Other'])
plt.show()
k=pd.crosstab(df.EDUCATION,df.DEFAULT,normalize='index')
print(k)
print(pd.crosstab(df.PAY_1,df.DEFAULT,margins=True))
sns.catplot(x='DEFAULT', y= 'LIMIT_BAL',jitter = True, data=df)
print(pd.crosstab(df.AGE,df.DEFAULT))
print(pd.crosstab(df.AGE,df.DEFAULT,normalize='index',margins=True))
sns.set(rc={'figure.figsize':(30,10)})
sns.set_context("talk", font_scale=0.7)
sns.heatmap(df.iloc[:,1:].corr(method='spearman'),
            cmap='rainbow_r', annot=True)
print(df.drop('DEFAULT', axis =1).apply(lambda x : x.corr(df.DEFAULT, method ='spearman')))

k=df.drop('DEFAULT', axis =1).columns 
print(k)
from scipy.stats import spearmanr
for index, col in enumerate(k):
    print(f'{col} : {round(spearmanr(df.DEFAULT, df[col])[0], 4)}')
    


