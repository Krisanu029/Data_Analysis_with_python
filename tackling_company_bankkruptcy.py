#to suppress warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#load the 5 raw .aiff files into a list
#attribute- relation file format(ARFF) file is an Ascii text file. it essentially provides a list of instances that commonly share an attribute set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#for loading .arff files
from scipy.io import arff

#load the 5 raw .arff files into list
def load_arff_raw_data():
    N=5
    return [arff.loadarff(str(i+1) + 'year.arff') for i in range(N)]
def load_dataframes():
    return [pd.DataFrame(data_i_year[0]) for data_i_year in load_arff_raw_data()]
def set_new_headers(dataframes):
    cols = ['X' + str(i +1) for i in range(len(dataframes[0].columns)-1)]
    cols.append('Y')
    for df in dataframes:
        df.columns = cols

dataframes = load_dataframes()
set_new_headers(dataframes)
print(dataframes[0].head())
print(dataframes[0].shape)

# convert the dtypes of all the columns to float
def convert_columns_type_float(dfs):
    for i in range(5):
        index =1
        while (index <= 63):
            colname = dfs[i].columns[index]
            col = getattr(dfs[i], colname)
            dfs[colname] = col.astype(float)
            index +=1
            convert_columns_type_float(dataframes)

def convert_class_label_type_int(dfs):
    for i in range(len(dfs)):
        col = getattr(dfs[i], 'Y')
        dfs[i]['Y'] = col.astype(int)
convert_class_label_type_int(dataframes)
print(dataframes[0].info())
'''
import pandas_profiling
for i in range(0,5):
    profile = dataframes[i].profile_report(title ='Pandas Profiling Report', plot ={'histogram' : {'bins': 8}})
    profile.to_file(output_file = str(i)+'output.html')
'''
import missingno as msno
msno.bar(dataframes[0], color='red', labels=True, sort ='ascending')
msno.bar(dataframes[1], color ='blue', labels = True, sort ='ascending')
msno.bar(dataframes[2], labels =True, sort ='ascending')

#imputation of missing value
#mean imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputed_df1= pd.DataFrame(imputer.fit_transform(dataframes[0]), columns = dataframes[0].columns)
msno.bar(mean_imputed_df1, color ='red', labels =True, sort ='ascending')
mean_imputed_df2 = pd.DataFrame(imputer.fit_transform(dataframes[1]), columns = dataframes[1].columns)
msno.bar(mean_imputed_df2, color ='blue', labels =True, sort ='ascending')
mean_imputed_df3 = pd.DataFrame(imputer.fit_transform(dataframes[2]), columns = dataframes[2].columns)
msno.bar(mean_imputed_df3, color ='blue', labels =True, sort ='ascending')
mean_imputed_df4 = pd.DataFrame(imputer.fit_transform(dataframes[3]), columns = dataframes[3].columns)
msno.bar(mean_imputed_df4, color ='blue', labels =True, sort ='ascending')
mean_imputed_df5 = pd.DataFrame(imputer.fit_transform(dataframes[4]), columns = dataframes[4].columns)
msno.bar(mean_imputed_df5, color ='blue', labels =True, sort ='ascending')

#iterative method
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()
iter_imputed_df1= pd.DataFrame(imputer.fit_transform(dataframes[0]), columns = dataframes[0].columns)
msno.bar(iter_imputed_df1, color ='red', labels =True, sort ='ascending')
iter_imputed_df2 = pd.DataFrame(imputer.fit_transform(dataframes[1]), columns = dataframes[1].columns)
msno.bar(iter_imputed_df2, color ='blue', labels =True, sort ='ascending')
iter_imputed_df3 = pd.DataFrame(imputer.fit_transform(dataframes[2]), columns = dataframes[2].columns)
msno.bar(iter_imputed_df3, color ='blue', labels =True, sort ='ascending')
iter_imputed_df4 = pd.DataFrame(imputer.fit_transform(dataframes[3]), columns = dataframes[3].columns)
msno.bar(iter_imputed_df4, color ='blue', labels =True, sort ='ascending')
iter_imputed_df5 = pd.DataFrame(imputer.fit_transform(dataframes[4]), columns = dataframes[4].columns)
msno.bar(iter_imputed_df5, color ='blue', labels =True, sort ='ascending')
#spliting target and features of dataframe
X0=mean_imputed_df1.drop('Y',axis=1)
y0=mean_imputed_df1.Y
#Second DataFrame
X1=mean_imputed_df2.drop('Y',axis=1)
y1=mean_imputed_df2.Y
#Third DataFrame
X2=mean_imputed_df3.drop('Y',axis=1)
y2=mean_imputed_df3.Y
X6=mean_imputed_df4.drop('Y',axis=1)
y6=mean_imputed_df4.Y
X7=mean_imputed_df5.drop('Y',axis=1)
y7=mean_imputed_df5.Y

#First DataFrame
X3=iter_imputed_df1.drop('Y',axis=1)
y3=iter_imputed_df1.Y
#Second DataFrame
X4=iter_imputed_df2.drop('Y',axis=1)
y4=iter_imputed_df2.Y
#Third DataFrame
X5=iter_imputed_df3.drop('Y',axis=1)
y5=iter_imputed_df3.Y
X8=iter_imputed_df4.drop('Y',axis=1)
y8=iter_imputed_df4.Y
X9=iter_imputed_df5.drop('Y',axis=1)
y9=iter_imputed_df5.Y

#feature selection with Lasso
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

feature_names = X0.columns.tolist()
lasso = Lasso(alpha =0.01,positive=True)
lasso.fit(X0,y0)
coef_list = sorted(map(lambda x: round(x,4), lasso.coef_.reshape(-1), feature_names), reverse=True)
print(coef_list[0:5])

features_names=X1.columns.tolist()
#We are initializing lasso 
lasso = Lasso(alpha=0.01 ,positive=True)
#We are fitting Lasso for X1 and y1
lasso.fit(X1,y1)
#We are getting the feature names after fitting Lasso 
coef_list=sorted(zip(map(lambda x: round(x,4), \
                     lasso.coef_.reshape(-1)), \
                     features_names),reverse=True)
print(coef_list [0:5])

features_names=X2.columns.tolist()
#We are initializing lasso 
lasso = Lasso(alpha=0.01 ,positive=True)
#We are fitting Lasso for X2 and y2
lasso.fit(X2,y2)
#We are getting the feature names after fitting Lasso 
coef_list=sorted(zip(map(lambda x: round(x,4), \
                     lasso.coef_.reshape(-1)), \
                     features_names),reverse=True)
print(coef_list [0:5])
    

