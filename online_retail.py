import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
r09 = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Analysis-Workshop/master/Chapter08/Datasets/online_retail_II.csv')
print(r09.head())
r10 = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Analysis-Workshop/master/Chapter08/Datasets/online_retail_II2.csv')
print(r10.head())
dfs =[r09, r10]
retail = pd.concat(dfs,keys =['09-10', '10-11'])
#print(retail)

retail.rename(index =str,
              columns ={
                  'Invoice' : 'invoice', 
    'StockCode' : 'stock_code', 
    'Quantity' : 'quantity', 
    'InvoiceDate' : 'date', 
    'Price' : 'unit_price', 
    'Country' : 'country', 
    'Description' : 'desc', 
    'Customer ID' : 'cust_id'}, inplace = True)

print(retail.head())

print(retail.isnull().sum().sort_values(ascending = False))

print(retail.describe())

print(retail[retail.unit_price == 38970.0])

print(retail.loc[retail.unit_price == -53594.36000])

print((retail.unit_price < 0).sum())

print((retail.quantity < 0).sum())

k =retail[(retail['unit_price'] <= 0) & (retail['quantity'] <= 0) & (retail['cust_id'].isnull())]

print(k)


null_retail = retail[retail.isnull().any(axis =1)]
print(null_retail)

new_retail = retail[(retail['unit_price'] > 0) & (retail['quantity'] > 0)]

print(new_retail.describe())

plt.figure(figsize =(12,6))
up = sns.boxplot(new_retail.unit_price)
plt.show()

plt.figure(figsize =(12,6))
q = sns.boxplot(new_retail.quantity)
plt.show()

new_retail = new_retail[new_retail.unit_price < 15000]
print(new_retail.describe())

plt.figure(figsize = (12,6))
up_new = sns.boxplot(new_retail.unit_price)
plt.show()

plt.figure(figsize = (12,6))
q = sns.boxplot(new_retail.quantity)
plt.show()

new_retail =new_retail[new_retail.quantity < 25000]
print(new_retail.describe())

plt.figure(figsize = (12,6))
q_new = sns.boxplot(new_retail.quantity)
plt.show()

k = new_retail[(new_retail.desc.isnull()) &(new_retail.cust_id.isnull())]
print(k)

print(new_retail.info())

new_retail = new_retail.dropna()

print(new_retail.info())

retail = new_retail
print(retail.head())
retail.desc = retail.desc.str.lower()
print(retail.head())

retail['date'] = pd.to_datetime(retail['date'], format = '%d/%m/%Y %H:%M')
print(retail.head())

retail.insert(loc = 4, column='year_month', value = retail.date.apply(lambda x: 100 * x.year + x.month))
retail.insert(loc =5, column ='year', value = retail.date.dt.year)
retail.insert(loc =6, column ='month', value = retail.date.dt.month)
retail.insert(loc =7, column ='day', value = retail.date.dt.day)
retail.insert(loc =8, column ='hour', value = retail.date.dt.hour)
retail.insert(loc = 9, column ='day_of_week', value = (retail.date.dt.dayofweek)+1)
print(retail.head())

retail.insert(loc = 11, column = 'spent', value =(retail['quantity']*retail['unit_price']))
retail = retail[['invoice', 'country', 'cust_id', 'stock_code', 'desc','quantity', 'unit_price', 'date', 'spent', 
                 'year_month', 'year', 'month', 'day', 'day_of_week', 'hour']]
print(retail.head())

#orders made by each customer
ord_cust = retail.groupby(['cust_id', 'country'], as_index = False)['invoice'].count()
print(ord_cust.head(10))

plt.subplots(figsize =(15,6))
oc = plt.plot(ord_cust.cust_id, ord_cust.invoice)
plt.xlabel('Customer ID')
plt.ylabel('Number of orders')
plt.title('Number of orders made by customer')
plt.show()

print(ord_cust.describe())
# 5 customers who ordered the most number of times
print(ord_cust.sort_values(by = 'invoice', ascending = False).head())

#money spent customer
spent_cust = retail.groupby(['cust_id', 'country'], as_index = False)['spent'].sum()
print(spent_cust.head())

plt.subplots(figsize = (15, 6))
sc = plt.plot(spent_cust.cust_id, spent_cust.spent)
plt.xlabel('Customer ID')
plt.ylabel('Total Amount Spent')
plt.title('Amount Spent by Customers')
plt.show()

# 5 customers who spent more money for retail product shopping
spent_cust.sort_values(by = 'spent', ascending = False).head()

print(retail.head())
print(retail.tail())

#orders per month
ord_month = retail.groupby(['invoice'])['year_month'].unique().value_counts().sort_index()
print(ord_month)

om = ord_month.plot(kind='bar', figsize = (15, 6))
om.set_xlabel('Month')
om.set_ylabel('Number of Orders')
om.set_title('Orders per Month')
om.set_xticklabels(('Dec 09', 'Jan 10', 'Feb 10', 'Mar 10', 'Apr 10', 'May 10', 
                           'Jun 10', 'Jul 10', 'Aug 10', 'Sep 10', 'Oct 10', 'Nov 10', 'Dec 10',
                   'Jan 11', 'Feb 11', 'Mar 11', 'Apr 11', 'May 11', 
                           'Jun 11', 'Jul 11', 'Aug 11', 'Sep 11', 'Oct 11', 'Nov 11', 'Dec 11'), rotation = 'horizontal')
plt.show()  

# most popular time of the month to order
ord_day = retail.groupby('invoice')['day'].unique().value_counts().sort_index()
print(ord_day)

od = ord_day.plot(kind='bar', figsize = (15, 6))
od.set_xlabel('Day of the Month')
od.set_ylabel('Number of Orders')
od.set_title('Orders per Day of the Month')
od.set_xticklabels(labels = [i for i in range (1, 32)], rotation = 'horizontal')
plt.show() 

#orders per  day of the week
ord_dayofweek = retail.groupby(['invoice'])['day_of_week'].unique().value_counts().sort_index()

odw = ord_dayofweek.plot(kind='bar', figsize = (15, 6))
odw.set_xlabel('Day of the Week')
odw.set_ylabel('Number of Orders')
odw.set_title('Orders per Day of the Week')
odw.set_xticklabels(labels = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'], rotation = 'horizontal')
plt.show()

q_item = retail.groupby(by = ['desc'], as_index = False)['quantity'].sum()
print(q_item.head())

print(q_item.sort_values(by = 'quantity', ascending = False).head())

item_month = retail.groupby(by = ['desc', 'year_month'], as_index = False)['quantity'].sum()
print(item_month.sort_values(by = 'quantity', ascending = False).head())

item_dayofweek = retail.groupby(by = ['desc', 'day_of_week'], as_index = False)['quantity'].sum()
print(item_dayofweek.sort_values(by = 'quantity', ascending = False).head())

item_coun = retail.groupby(by = ['desc', 'country'], as_index = False)['quantity'].sum()
print(item_coun.sort_values(by = 'quantity', ascending = False).head())

retail_sort = retail.sort_values(['cust_id', 'stock_code', 'date'])
retail_sort_shift1 = retail_sort.shift(1)
retail_sort_reorder = retail_sort.copy()

retail_sort_reorder['reorder'] = np.where(retail_sort['stock_code'] == retail_sort_shift1['stock_code'], 1, 0)
print(retail_sort_reorder.head())

print(retail_sort_shift1.head())

rsr = pd.DataFrame((retail_sort_reorder.groupby('desc')['reorder'].sum())).sort_values('reorder', ascending = False)
print(rsr.head())

q_up = retail.groupby(by = ['unit_price'], as_index = False)['quantity'].sum()
print(q_up.sort_values('quantity', ascending = False).head(10))

up_arr = np.array(retail.unit_price)
q_arr = np.array(retail.quantity)

print(np.corrcoef(up_arr, q_arr))

ord_coun = retail.groupby(['country'])['invoice'].count().sort_values()
print(ord_coun.head())

ocoun = ord_coun.plot(kind='barh', figsize = (15, 6))
ocoun.set_xlabel('Number of Orders')
ocoun.set_ylabel('Country')
ocoun.set_title('Orders per Country')
plt.show() 

del ord_coun['United Kingdom']

ocoun2 = ord_coun.plot(kind='barh', figsize = (15, 6))
ocoun2.set_xlabel('Number of Orders')
ocoun2.set_ylabel('Country')
ocoun2.set_title('Orders per Country')
plt.show() 

coun_spent = retail.groupby('country')['spent'].sum().sort_values()

cs = coun_spent.plot(kind='barh', figsize = (15, 6))
cs.set_xlabel('Amount Spent')
cs.set_ylabel('Country')
cs.set_title('Amount Spent per Country')
plt.show() 

del coun_spent['United Kingdom']

cs2 = coun_spent.plot(kind='barh', figsize = (15, 6))
cs2.set_xlabel('Amount Spent')
cs2.set_ylabel('Country')
cs2.set_title('Amount Spent per Country')
plt.show() 









