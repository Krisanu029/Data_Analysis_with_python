import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Analysis-Workshop/master/Chapter09/Datasets/energydata_complete.csv')

print(data.head())

print(data.columns)
print(data.index)
#let's check whether the data contains null value or not
print(data.isnull().sum())

#rename the columns

df1 = data.rename(columns = {
    'date' : 'date_time', 
    'Appliances' : 'a_energy', 
    'lights' : 'l_energy', 
    'T1' : 'kitchen_temp', 
    'RH_1' : 'kitchen_hum', 
    'T2' : 'liv_temp', 
    'RH_2' : 'liv_hum', 
    'T3' : 'laun_temp', 
    'RH_3' : 'laun_hum', 
    'T4' : 'off_temp', 
    'RH_4' : 'off_hum', 
    'T5' : 'bath_temp', 
    'RH_5' : 'bath_hum', 
    'T6' : 'out_b_temp', 
    'RH_6' : 'out_b_hum', 
    'T7' : 'iron_temp', 
    'RH_7' : 'iron_hum', 
    'T8' : 'teen_temp', 
    'RH_8' : 'teen_hum', 
    'T9' : 'par_temp', 
    'RH_9' : 'par_hum',
    'T_out' : 'out_temp',
    'Press_mm_hg' : 'out_press',
    'RH_out' : 'out_hum',
    'Windspeed' : 'wind',
    'Visibility' : 'visibility',
    'Tdewpoint' : 'dew_point',
    'rv1' : 'rv1',
    'rv2' : 'rv2'
})

print(df1.head())

print(df1.tail())

print(df1.describe())

lights_box = sns.boxplot(df1.l_energy)

l= [0, 10, 20, 30, 40, 50, 60, 70]
counts =[]

for i in l:
    a= (df1.l_energy == i).sum()
    counts.append(a)

print(counts)

lights = sns.barplot(x= l, y=counts)
lights.set_xlabel('energy consumed by light')
lights.set_ylabel('Number of lights')
lights.set_title('Distribution of energy consumed by light')
plt.show()

print((df1.l_energy == 0).sum()/(df1.shape[0])*100)

new_data = df1

new_data.drop(['l_energy'], inplace= True, axis =1)

print(new_data.head())

app_box = sns.boxplot(new_data.a_energy)

out = (new_data.a_energy > 200).sum()
print(out)

out_e = (new_data['a_energy'] > 950).sum()
print(out_e)

energy = new_data[(new_data['a_energy'] <= 200)]

print(energy.describe())

new_en = energy

new_en['date_time'] = pd.to_datetime(new_en.date_time, format = '%Y-%m-%d %H:%M:%S')

print(new_en.head())

new_en.insert(loc =1, column ='month', value = new_en.date_time.dt.month)

new_en.insert(loc =2, column ='day', value = (new_en.date_time.dt.dayofweek)+1)

print(new_en.head())

import plotly.graph_objs as go
plt.figure(figsize =(15, 20))
app_date = go.Scatter(x = new_en.date_time, mode ='lines', y = new_en.a_energy)

layout = go.Layout(title = 'Appliance Energy Consumed by Date', xaxis = dict(title='Date'), yaxis = dict(title='Wh'))
fig = go.Figure(data =[app_date], layout = layout)
fig.show()

app_mon = new_en.groupby(by='month', as_index=False)['a_energy'].sum()
print(app_mon)

print(app_mon.sort_values(by ='a_energy', ascending =False).head())

plt.subplots(figsize = (15, 6))
am = sns.barplot(app_mon.month, app_mon.a_energy)
plt.xlabel('Month')
plt.ylabel('Energy Consumed by Appliances')
plt.title('Total Energy Consumed by Appliances per Month')
plt.show()

app_day = new_en.groupby(by='day', as_index=False)['a_energy'].sum()
print(app_day)

print(app_day.sort_values(by ='a_energy', ascending = False).head())

plt.subplots(figsize = (15, 6))
ad = sns.barplot(app_day.day, app_day.a_energy)
plt.xlabel('Day of the Week')
plt.ylabel('Energy Consumed by Appliances')
plt.title('Total Energy Consumed by Appliances')
plt.show()

col_temp = ['kitchen_temp', 'liv_temp', 'laun_temp', 'off_temp', 
'bath_temp', 'out_b_temp', 'iron_temp', 'teen_temp', 'par_temp']

temp = new_en[col_temp]

print(temp.head())

temp.hist(bins =15, figsize=(12,16))

col_hum = ['kitchen_hum', 'liv_hum', 'laun_hum', 'off_hum', 'bath_hum', 'out_b_hum', 'iron_hum', 'teen_hum', 'par_hum']
hum = new_en[col_hum]
print(hum.head())

hum.hist(bins = 15, figsize = (12, 16))

col_weather = ["out_temp", "dew_point", "out_hum", "out_press",
                "wind", "visibility"] 
weath = new_en[col_weather]
weath.head()
weath.hist(bins = 15, figsize = (12, 16))


