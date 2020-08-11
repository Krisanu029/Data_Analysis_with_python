import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
hourly_data = pd.read_csv('https://raw.githubusercontent.com/'\
                          'PacktWorkshops/'\
                          'The-Data-Analysis-Workshop/'\
                          'master/Chapter01/data/hour.csv')
print(hourly_data.head())
print(hourly_data.columns)
print(f'shape of data : {hourly_data.shape}')
print(f'number of missing values in the data: {hourly_data.isnull().sum().sum()}')
print(hourly_data.describe().T)

preprocessed_data = hourly_data.copy()

season_mapping = {1 : 'winter', 2 : 'spring', 3 : 'summer', 4 : 'fall'}
preprocessed_data['season'] = preprocessed_data['season'].map(season_mapping)
print(preprocessed_data['season'].value_counts())
yr_mapping = {0 : 2011, 1 : 2012}
preprocessed_data['yr'] = preprocessed_data['yr'].apply(lambda x : yr_mapping[x])
print(preprocessed_data['yr'].value_counts())
weekday_mapping = {0 : 'Sunday', 1 : 'Monday', 2 : 'Tuesday', 3 : 'Wednesday',
                   4 : 'Thursday', 5 : 'Friday', 6 : 'Saturday'}
preprocessed_data['weekday'] = preprocessed_data['weekday'].map(weekday_mapping)
print(preprocessed_data['weekday'].value_counts())
weather_mapping ={1 : 'clear', 2 : 'cloudy', 3 : 'light_rain_show', 4 : 'heavy_rain_show'}
preprocessed_data['weathersit'] = preprocessed_data['weathersit'].apply(lambda x : weather_mapping[x])
print(preprocessed_data['weathersit'].value_counts())
preprocessed_data['hum'] = preprocessed_data['hum'] * 100
preprocessed_data['windspeed'] = preprocessed_data['windspeed'] * 67
cols = ['season', 'yr', 'weekday', 'weathersit', 'hum', 'windspeed']
preprocessed_data[cols].sample(10,random_state =123)
assert (preprocessed_data.casual + preprocessed_data.registered == preprocessed_data.cnt).all(),'sum of casual and registered rides not equal to total number of rides'

'''
#rides distribution between casual and registered riders
sns.distplot(preprocessed_data['casual'], label = 'casual')
sns.distplot(preprocessed_data['registered'], label ='registered')
plt.legend()
plt.xlabel('rides', fontsize = 12)
plt.title('rides distribution', fontsize = 12)
plt.savefig('rides_distribution.png', format = 'png')
plt.show()

#plot evolution of rides over the time
plot_data = preprocessed_data[['registered', 'casual', 'dteday']]
plt.figure(figsize=(10, 6))
data = plot_data.groupby('dteday').sum()
plt.plot(data)
plt.xlabel('time')
plt.ylabel('number of rides per day')
plt.show()
plt.savefig('rides_daily.png', format='png')

#ploting the rolling mean or moving average over 1 week period
plot_data = preprocessed_data[['registered','casual', 'dteday']]
plot_data = plot_data.groupby('dteday').sum()
window = 7
rolling_means = plot_data.rolling(window).mean()
rolling_std = plot_data.rolling(window).std()
ax = rolling_means.plot(figsize=(10, 6))
ax.fill_between(rolling_means.index, rolling_means['registered'] + 2*rolling_std['registered'],
                rolling_means['registered']- 2*rolling_std['registered'], alpha =0.2)
ax.fill_between(rolling_means.index, rolling_means['casual']+2*rolling_std['casual'],
                rolling_means['casual']- 2*rolling_std['casual'], alpha = 0.2)
plt.xlabel('time')
plt.ylabel('number of rides per day')
plt.legend(loc ='upper_left')
plt.show()
plt.savefig('rides_aggregated.png', format ='png') 

plot_data = preprocessed_data[['hr', 'weekday', 'registered', 'casual']]
plot_data = plot_data.melt(id_vars= ['hr', 'weekday'], var_name = 'type', value_name ='count')
grid = sns.FacetGrid(plot_data, row = 'weekday',col='type', height=2.5,aspect=2.5,
                     row_order=['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'])
grid.map(sns.barplot, 'hr', 'count', alpha =0.5)
grid.savefig('weekday_hour_distributions.png', format = 'png')


#analyzing seasonal impact on rides
plot_data =preprocessed_data[['hr', 'season', 'registered', 'casual']]
plot_data = plot_data.melt(id_vars =['hr', 'season'], var_name ='type', value_name ='count')
grid=sns.FacetGrid(plot_data, row = 'season', col = 'type', height=2.5, aspect=2.5,
              row_order= ['winter', 'spring', 'summer', 'fall'])

grid.map(sns.barplot, 'hr','count', alpha =0.5)
grid.savefig('season_hr_distribution.png', format ='png')

plot_data = preprocessed_data[['weekday', 'season', 'registered', 'casual']]
plot_data = plot_data.melt(id_vars = ['weekday', 'season'], var_name = 'type', value_name = 'count')
grid = sns.FacetGrid(plot_data,row = 'season', col= 'type',height = 2.5, aspect = 2.5,
                     row_order =['winter', 'spring', 'summer', 'fall'])
grid.map(sns.barplot, 'weekday', 'count', alpha =0.5)
grid.savefig('weekday_season_distribution.png', format ='png')

#hypothesis testing
#----------------------
population_mean = preprocessed_data.registered.mean()
sample = preprocessed_data[(preprocessed_data.season =='summer')& (preprocessed_data.yr ==2011) ].registered
from scipy.stats import ttest_1samp
test_result = ttest_1samp(sample, population_mean)
print(f'test_statstics : {test_result[0]}, p_value: {test_result[1]}')

import random
random.seed(111)
sample_unbiased = preprocessed_data.registered.sample(frac =0.05)
test_unbiased = ttest_1samp(sample_unbiased, population_mean)
print(f'unbiased : test_statistics : {test_unbiased[0]}, p_value : {test_unbiased[1]}')

from scipy.stats import ttest_ind
weekend_days = ['Saturday', 'Sunday']
weekend_mask = preprocessed_data.weekday.isin(weekend_days)
workingdays_mask = ~preprocessed_data.weekday.isin(weekend_days)
weekend_data = preprocessed_data.registered[weekend_mask]
workingdays_data = preprocessed_data.registered[workingdays_mask]
test_res = ttest_ind(weekend_data, workingdays_data)
print(f'test_statistics : {test_res[0]:0.3f}, p_value : {test_res[1]:0.3f}')
sns.distplot(weekend_data, label ='weekend days')
sns.distplot(workingdays_data, label ='working days')
plt.legend()
plt.xlabel('rides')
plt.ylabel('frequency')
plt.title('Registered rides distribution')
plt.show()

from scipy.stats import ttest_ind
weekend_days = ['Saturday', 'Sunday']
weekend_mask = preprocessed_data.weekday.isin(weekend_days)
workingdays_mask = ~preprocessed_data.weekday.isin(weekend_days)
weekend_data = preprocessed_data.casual[weekend_mask]
workingdays_data = preprocessed_data.casual[workingdays_mask]
test_res = ttest_ind(weekend_data, workingdays_data)
print(f'test_statistics : {test_res[0]:0.3f}, p_value : {test_res[1]:0.3f}')
sns.distplot(weekend_data, label ='weekend days')
sns.distplot(workingdays_data, label ='working days')
plt.legend()
plt.xlabel('rides')
plt.ylabel('frequency')
plt.title('casual rides distribution')
plt.show()

def plot_correlations(data, col):
    corr_r = np.corrcoef(data[col], data['registered'])[0,1]
    ax = sns.regplot(x = col, y = 'registered', data = data, scatter_kws = {'alpha' : 0.05},label=f'Registered rides (correlation: {corr_r:.3f})')
    corr_c = np.corrcoef(data[col], data['casual'])[0,1]
    ax = sns.regplot(x=col, y= 'casual',data=data, scatter_kws={'alpha':0.05}, label =f'casual rides (correlation: {corr_c:.3f})')
    legend = ax.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(0.5)
    ax.set_ylabel('rides')
    ax.set_title(f'correlation coefficient between rides and {col}')
    return ax
plt.figure(figsize=(10,8))
ax = plot_correlations(preprocessed_data, 'temp')
plt.savefig('correlation_temp.png', format ='png')
plt.figure(figsize =(10,8))
ax = plot_correlations(preprocessed_data, 'atemp')
plt.savefig('correlation_atemp.png',format ='png')
plt.figure(figsize =(10,8))
ax =plot_correlations(preprocessed_data, 'hum')
plt.savefig('correlation_hum.png', format ='png')
plt.figure(figsize =(10,8))
ax = plot_correlations(preprocessed_data,'windspeed')
plt.savefig('correlation_windspeed.png', format ='png')

x= np.linspace(0,5,100)
y_lin = 0.5*x + 0.1 * np.random.randn(100)
y_exp =np.exp(x) + 0.1 * np.random.randn(100)
from scipy.stats import pearsonr, spearmanr
corr_lin_pearson = pearsonr(x, y_lin)[0]
corr_lin_spearman = spearmanr(x, y_lin)[0]
corr_exp_pearson = pearsonr(x, y_exp)[0]
corr_exp_spearman = spearmanr(x, y_exp)[0]
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.scatter(x, y_lin)
ax1.set_title(f'Linear relationship\nPearson:{corr_lin_pearson:.3f}, Spearman:{corr_lin_spearman:.3f}')
ax2.scatter(x,y_exp)
ax2.set_title(f'Monotonic relationship\nPearson:{corr_exp_pearson:.3f}, Searman: {corr_exp_spearman:.3f}')
plt.show()

from scipy.stats import pearsonr, spearmanr
def compute_correlations(data, col):
    pearson_reg = pearsonr(data[col], data['registered'])[0]
    pearson_cas =pearsonr(data[col], data['casual'])[0]
    spearman_reg =spearmanr(data[col], data['registered'])[0]
    spearman_cas = spearmanr(data[col], data['casual'])[0]
    return pd.Series({'Pearson(registered)': pearson_reg, 
                     'Spearman(registered)':spearman_reg,
                     'Pearson(casual)': pearson_cas,
                     'Spearman(casual)':spearman_cas})
cols = ['temp', 'atemp', 'hum', 'windspeed']
corr_data = pd.DataFrame(index= ['Pearson(registered)','Spearman(registered)','Pearson(casual)','Spearman(casual)'])
for col in cols:
    corr_data[col] = compute_correlations(preprocessed_data, col)
print(corr_data)
#plot correlation matrix
cols = ['temp', 'atemp', 'hum', 'windspeed','registered', 'casual']
plot_data = preprocessed_data[cols]
corr = plot_data.corr()
fig = plt.figure(figsize=(10,8))
plt.matshow(corr, fignum=fig.number)
plt.xticks(range(len(plot_data.columns)), plot_data.columns)
plt.yticks(range(len(plot_data.columns)), plot_data.columns)
plt.colorbar()
plt.ylim([5.5, -0.5])
plt.show()
'''
from statsmodels.tsa.stattools import adfuller

def test_stationary(ts, window=10, **kwargs):
    plot_data = pd.DataFrame(ts)
    plot_data['rolling_mean'] = ts.rolling(window).mean()
    plot_data['rolling_std'] = ts.rolling(window).std()
    p_value = adfuller(ts)[1]
    ax= plot_data.plot(**kwargs)
    ax.set_title(f'Dickey-Fuller p-value: {p_value:.3f}')

#get daily rides
daily_rides = preprocessed_data[['dteday', 'registered', 'casual']]
daily_rides = daily_rides.groupby('dteday').sum()
daily_rides.index = pd.to_datetime(daily_rides.index)
#test_stationary(daily_rides['registered'], figsize=(10,8))
#test_stationary(daily_rides['casual'],figsize =(10,8))
#print(daily_rides)

#sutract rolling mean
registered = daily_rides['registered']
registered_ma = registered.rolling(10).mean()
registered_ma_diff = registered - registered_ma
registered_ma_diff.dropna(inplace =True)

casual = daily_rides['casual']
casual_ma = casual.rolling(10).mean()
casual_ma_diff = casual - casual_ma
casual_ma_diff.dropna(inplace =True)
'''
plt.figure()
test_stationary(registered_ma_diff, figsize =(10,8))

plt.figure()
test_stationary(casual_ma_diff, figsize =(10,8))
'''
registered = daily_rides["registered"]
registered_diff = registered - registered.shift()
registered_diff.dropna(inplace=True)
casual = daily_rides["casual"]
casual_diff = casual - casual.shift()
casual_diff.dropna(inplace=True)
'''
plt.figure()
test_stationary(registered_diff, figsize=(10, 8))

plt.figure()
test_stationary(casual_diff, figsize=(10, 8))
'''
from statsmodels.tsa.seasonal import seasonal_decompose
registered_decomposition = seasonal_decompose(daily_rides['registered'])
casual_decomposition = seasonal_decompose(daily_rides['casual'])
'''
registered_plot = registered_decomposition.plot()
registered_plot.set_size_inches(10,8)
casual_plot = casual_decomposition.plot()
casual_plot.set_size_inches(10,8)
'''
plt.figure()
test_stationary(registered_decomposition.resid.dropna(), figsize =(10,8))

plt.figure()
test_stationary(casual_decomposition.resid.dropna(), figsize = (10,8))

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(3,3, figsize =(25,12))
original = daily_rides['registered']
axes[0,0].plot(original)
axes[0,0].set_title('original series')
plot_acf(original, ax = axes[0,1])
plot_pacf(original, ax = axes[0,2])
first_order_int = original.diff().dropna()
axes[1,0].plot(first_order_int)
axes[1,0].set_title("First order integrated")
plot_acf(first_order_int, ax=axes[1,1])
plot_pacf(first_order_int, ax=axes[1,2])
# plot first order integrated series
second_order_int = first_order_int.diff().dropna()
axes[2,0].plot(first_order_int)
axes[2,0].set_title("Second order integrated")
plot_acf(second_order_int, ax=axes[2,1])
plot_pacf(second_order_int, ax=axes[2,2])

from pmdarima import auto_arima
model = auto_arima(registered, start_p=1, start_q=1, max_p=3, max_q=3, information_criterion="aic")
print(model.summary())
plot_data = pd.DataFrame(registered)
plot_data['predicted'] = model.predict_in_sample()
plot_data.plot(figsize=(12, 8))
plt.ylabel("number of registered rides")
plt.title("Predicted vs actual number of rides")
plt.savefig('figs/registered_arima_fit.png', format='png')  


        




