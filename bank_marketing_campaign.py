import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
bank_data = pd.read_csv("https://raw.githubusercontent.com/"\
                        "PacktWorkshops/"\
                        "The-Data-Analysis-Workshop/"\
                        "master/Chapter03/data/bank-additional/"\
                        "bank-additional-full.csv", sep =';')
print(bank_data.head())

#define the numerical features

numerical_features = [ col for col in bank_data.columns if np.issubdtype(bank_data[col].dtype, np.number)]

print(numerical_features)

#print statistics about the different numerical features

print(bank_data[numerical_features].describe().T)

plt.figure(figsize=(10,18))

for index, col in enumerate(numerical_features):
    plt.subplot(5, 2, index+1)
    sns.distplot(bank_data[col], kde_kws={'bw': 0.1})
plt.savefig('distributions_numerical_feature.png', format ='png', dpi =500)

categorical_features = [col for col in bank_data.columns if pd.api.types.is_string_dtype(bank_data[col])]
print(categorical_features)

plt.figure(figsize =(25,35))
for index, col in enumerate(categorical_features):
    plt.subplot(6, 2, index +1)
    ax= sns.countplot(x=col, data= bank_data)
    ax.set_xlabel('count')
    ax.set_ylabel(col)
    ax.tick_params(labelsize =20)
    plt.show()
print('Total number of entries')
print(bank_data['y'].value_counts(ascending =True))
print('Percentage:')
print(bank_data['y'].value_counts(normalize =True, ascending =True)*100)
plt.figure(figsize=(10,18))
for index, col in enumerate(numerical_features):
    plt.subplot(5,2,index+1)
    sns.violinplot(x=col, y='y', data = bank_data, order =['Yes','No'])
plt.show()
from scipy.stats import ttest_ind

def test_means(data, col):
    yes_mask = data['y'] =='yes'
    values_yes = data[col][yes_mask]
    values_no = data[col][~yes_mask]
    mean_yes = values_yes.mean()
    mean_no = values_no.mean()
    ttest_res = ttest_ind(values_yes, values_no)
    return [col, mean_yes, mean_no, round(ttest_res[0],4), round(ttest_res[1],4)]
test_df = pd.DataFrame(columns = ['column', 'mean yes', 'mean no', 'ttest stat', 'ttest pval'])
for index, col in   enumerate(numerical_features):
    test_df.loc[index] = test_means(bank_data, col)
print(test_df)

#finding th equality of diztribution
from scipy.stats import ks_2samp

def test_ks(data, col):
    yes_mask = data['y'] =='yes'
    values_yes =data[col][yes_mask]
    values_no =  data[col][~yes_mask]
    kstest_res = ks_2samp(values_yes, values_no)
    return [col, round(kstest_res[0], 4), round(kstest_res[1], 4)]

test_df = pd.DataFrame(columns =['column', 'kstest stats', 'kstest p_value'])

for index, col in enumerate(numerical_features):
    test_df.loc[index] = test_ks(bank_data, col)
print(test_df)

#create arrays containing campaign and financial columns
campaign_columns = ['age', 'duration', 'campaign','previous']
financial_columns =['emp.var.rate','cons.price.idx', 'cons.conf.idx', 'euribor3m']

#create pairplot between campaign columns
plot_data= bank_data[campaign_columns +['y']]
plt.figure(figsize=(10,10))
sns.pairplot(data= plot_data, hue='y', palette='bright', diag_kws={'bw':0.1})
plt.show()

#create pairplot between financial columns
plot_data = bank_data[financial_columns +['y']]
plt.figure(figsize =(10,10))
sns.pairplot(data= plot_data, hue ='y', palette='bright', diag_kws={'bw':0.1})

plt.show()

#creating correlation matrix for successful calls
successful_calls = bank_data['y'] == 'yes'
plot_data = bank_data[campaign_columns + financial_columns][successful_calls]
successful_corr = plot_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(successful_corr,annot =True, cmap ='coolwarm')
plt.show()

#plot correlation matrix plot for unsuccessful calls
plot_data = bank_data[campaign_columns + financial_columns][~successful_calls]
unsuccessful_corr = plot_data.corr()
plt.figure(figsize =(10,10))
sns.heatmap(unsuccessful_corr, annot =True, cmap = 'coolwarm')
plt.show()

diff_corr = successful_corr - unsuccessful_corr
plt.figure(figsize =(10,10))
sns.heatmap(diff_corr, annot = True, cmap ='coolwarm')
plt.show()

import statsmodels.api as sm
x = bank_data[['emp.var.rate', 'cons.price.idx', 'euribor3m']]
x= sm.add_constant(x)
y = bank_data['cons.conf.idx']
linear_regression_model = sm.OLS(y,x)
result = linear_regression_model.fit()
print(result.summary())

#plot logit function
import numpy as np
import matplotlib.pyplot as plt
x= np.arange(0,1,0.01)
logit = np.log(x/(1-x))
plt.figure(figsize =(6,6))
plt.plot(x,logit)
plt.xlabel('p')
plt.ylabel('$\log(\\frac{p}{1-p})$')
plt.grid()
plt.show()

x = bank_data[['age', 'duration', 'campaign', 'previous']]
x = sm.add_constant(x)
y = np.where(bank_data['y'] == 'yes', 1, 0)
logistic_regression_model = sm.Logit(y, x)
result = logistic_regression_model.fit()
print(result.summary())
    
y = np.where(bank_data['y'] =='yes',1,0)
print(bank_data['education'].unique())

hot_encoded = pd.get_dummies(bank_data['education'])
hot_encoded['education'] =  bank_data['education']
print(hot_encoded.head(10))

#transform all features into numerical ones, by using the get_dummies
x= bank_data.drop('y', axis =1)
x= pd.get_dummies(x)
x= sm.add_constant(x)
print(x.columns)

#extract and transform target variable into binary format
y = np.where(bank_data['y'] == 'yes', 1, 0)
#define the model and fit the model on our data
full_logistic_regression_model = sm.Logit(y, x)
result = full_logistic_regression_model.fit(maxiter = 500)
print(result.summary())




