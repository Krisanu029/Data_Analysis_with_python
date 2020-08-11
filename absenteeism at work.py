import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('https://raw.githubusercontent.com'\
                   '/PacktWorkshops/The-Data-Analysis-Workshop'\
                   '/master/Chapter02/data/'\
                   'Absenteeism_at_work.csv',sep =';')
print(data.head(10))
print(f'Data dimension: {data.shape}')
for col in data.columns:
    print(f'Column: {col:35} | type : {str(data[col].dtype):7} | missing values: {data[col].isna().sum():2d}')

print(data.describe().T)
#month encoding dictionaries
month_encoding = {1 : 'January', 2 : 'February', 3 : 'March', 4: 'April',
                  5 : 'May', 6 : 'June', 7 : 'July', 8 : 'August',
                  9 : 'September', 10 : 'October', 11 : 'November', 12 : 'December', 0 : 'Unknown'}
dow_encoding = {2: "Monday", 3: "Tuesday", 4: "Wednesday", \
                5: "Thursday", 6: "Friday"}
season_encoding = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
education_encoding = {1: "high_school", 2: "graduate", \
                      3: "postgraduate", 4: "master_phd"}
yes_no_encoding = {0: "No", 1: "Yes"}

preprocessed_data = data.copy()

preprocessed_data['Month of absence']= preprocessed_data['Month of absence'].map(month_encoding)

preprocessed_data['Day of the week'] = preprocessed_data['Day of the week'].map(dow_encoding)

preprocessed_data['Seasons'] = preprocessed_data['Seasons'].apply(lambda x : season_encoding[x])

preprocessed_data['Education'] = preprocessed_data['Education'].map(education_encoding)

preprocessed_data['Disciplinary failure'] = preprocessed_data['Disciplinary failure'].map(yes_no_encoding)

preprocessed_data['Social smoker'] = preprocessed_data['Social smoker'].map(yes_no_encoding)

preprocessed_data['Social drinker'] = preprocessed_data['Social drinker'].map(yes_no_encoding)

print(preprocessed_data.head().T)

#identifying the Reason for absence(ICD = international code of diseases)

def in_icd(val):
    return 'Yes' if val >1 and val <= 21 else 'No'

preprocessed_data['Disease'] = preprocessed_data['Reason for absence'].apply(in_icd)

plt.figure(figsize = (10,8))

sns.countplot(x='Disease', data= preprocessed_data)

plt.savefig('disease_plot.png', format = 'png', dpi = 300)

#initial analysis of the reason for absence

plt.figure(figsize = (10,5))

ax = sns.countplot(x='Reason for absence', data = preprocessed_data, hue = 'Disease', hue_order=['Yes', 'No'])

ax.set_ylabel('Number of entries for reason of absence')

plt.savefig('reason_absence_distribution.png', format = 'png', dpi = 300)

#plot reasons for absence against being a social drinker/smoker
plt.figure(figsize =(8,6))

sns.countplot(x = 'Reason for absence', data = preprocessed_data, hue = 'Social drinker', hue_order =['Yes', 'No'])

plt.savefig('reason_absence_drinker.png', format = 'png', dpi = 300)

plt.figure(figsize = (8,6))

sns.countplot(x='Reason for absence', data = preprocessed_data, hue = 'Social smoker', hue_order=['Yes', 'No'])

plt.savefig('reason_absence_smoker.png', format ='png', dpi = 300)

print(preprocessed_data['Social drinker'].value_counts(normalize =True))

print(preprocessed_data['Social smoker'].value_counts(normalize=True))

'''
#conditional probability
sample_space = set(['BB', 'BG', 'GB', 'GG'])

event_a = set(['BB'])

event_b = set(['BG', 'GB', 'BB'])

cond_prob = (0.25 * len(event_a.intersection(event_b)))/ (0.25 * len(event_b))

print(round(cond_prob, 4))
'''
#compute probabilites of being a drinker and smoker

drinker_prob = preprocessed_data['Social drinker'].value_counts(normalize = True)['Yes']

smoker_prob = preprocessed_data['Social smoker'].value_counts(normalize = True)['Yes']

print(f'P(drinker_prob) : {drinker_prob:.3f} | p(smoker_prob) : {smoker_prob:.3f}')

#create mask for drinker and smoker
drinker_mask = preprocessed_data['Social drinker'] == 'Yes'

smoker_mask = preprocessed_data['Social smoker'] == 'Yes'

total_entries = preprocessed_data.shape[0]

absence_drinker_prob = preprocessed_data['Reason for absence'][drinker_mask].value_counts()/total_entries

absence_smoker_prob = preprocessed_data['Reason for absence'][smoker_mask].value_counts()/total_entries

#compute conditional probabilities

cond_prob = pd.DataFrame(index = range(0, 29))

cond_prob['P(Absence | Social drinker)'] = absence_drinker_prob/drinker_prob

cond_prob['P(Absence | Social smoker'] = absence_smoker_prob/smoker_prob

plt.figure()

ax = cond_prob.plot.bar(figsize=(10,6))

ax.set_ylabel('Conditional probability')

plt.savefig('conditional_prob.png', format = 'png', dpi = 300)

#compute reason for absence probabilities
absence_prob = preprocessed_data['Reason for absence'].value_counts(normalize = True)

cond_prob_drinker_smoker = pd.DataFrame(index=range(0,29))
cond_prob_drinker_smoker["P(Social drinker | Absence)"] = \
cond_prob['P(Absence | Social drinker)']*drinker_prob/absence_prob
cond_prob_drinker_smoker["P(Social smoker | Absence)"] = \
cond_prob['P(Absence | Social smoker']*smoker_prob/absence_prob
plt.figure()
ax = cond_prob_drinker_smoker.plot.bar(figsize=(10,6))
ax.set_ylabel("Conditional probability")
plt.savefig('conditional_probabilities_drinker_smoker.png', \
            format='png', dpi=300)

# create violin plots of the absenteeism time in hours
plt.figure(figsize = (8,6))

sns.violinplot(x='Social drinker', y= 'Absenteeism time in hours', data = preprocessed_data, order=['No','Yes'])

plt.show()

plt.figure(figsize = (8,6))

sns.violinplot(x='Social smoker', y= 'Absenteeism time in hours', data = preprocessed_data, order=['No','Yes'])

plt.show()

from scipy.stats import ttest_ind

hours_col = 'Absenteeism time in hours'

drinker_mask = preprocessed_data['Social drinker'] =='Yes'

hours_drinkers = preprocessed_data.loc[drinker_mask, hours_col]
hours_non_drinker = preprocessed_data.loc[~drinker_mask, hours_col]

drinker_test =ttest_ind(hours_drinkers, hours_non_drinker)

print(f'statiscal value: {drinker_test[0]}, p-value : {drinker_test[1]}')

smoker_mask = preprocessed_data['Social smoker'] == 'Yes'

hours_smoker = preprocessed_data.loc[smoker_mask, hours_col]

hours_non_smoker = preprocessed_data.loc[~smoker_mask, hours_col]

smoker_test = ttest_ind(hours_smoker, hours_non_smoker)

print(f'statical value : {smoker_test[0]}, p_value : {smoker_test[1]}')

#perform kolmogorov-smirnov test for comparing distributions
from scipy.stats import ks_2samp

ks_drinker = ks_2samp(hours_drinkers,hours_non_drinker)

ks_smoker = ks_2samp(hours_smoker, hours_non_smoker)
print(f"Drinkers comparison: statistics={ks_drinker[0]:.3f}, \
pvalue={ks_drinker[1]:.3f}")
print(f"Smokers comparison:  statistics={ks_smoker[0]:.3f}, \
pvalue={ks_smoker[1]:.3f}")

#define function for computing the BMI category, based on BMI value

def get_bmi_category(bmi):
    if bmi < 18.5 :
        category = 'underweight'
    elif bmi >= 18.5 and bmi < 25 :
        category = 'healthy weight'
    elif bmi >= 25 and bmi < 30:
        category ='overweight'
    else:
        category = 'obese'
    return category

preprocessed_data['BMI category'] = preprocessed_data['Body mass index'].apply(get_bmi_category)

plt.figure(figsize = (10, 6))

sns.countplot(x='BMI category', data = preprocessed_data, order = ['underweight','healthy weight',
                                                                   'overweight','obese'],palette='Set2')
plt.show()

#plot BMI categories vs Reason for absence
plt.figure(figsize = (10, 16))

ax = sns.countplot(x = 'Reason for absence', hue = 'BMI category', data = preprocessed_data,
                   hue_order= ['underweight', 'healthy weight', 'overweight', 'obese'], palette='Set2')
ax.set_xlabel('number of employees')
plt.show()

#plot   distributon of absence time based on BMI category
plt.figure(figsize =(8,6))

sns.violinplot(x = 'BMI category', y = 'Absenteeism time in hours', data = preprocessed_data,
               order = ['healthy weight', 'overweight', 'obese'])
plt.show()

from scipy.stats import pearsonr

pearson_test = pearsonr(preprocessed_data['Age'], preprocessed_data['Absenteeism time in hours'])

plt.figure(figsize = (10,6))

ax = sns.regplot(x ='Age', y = 'Absenteeism time in hours', data = preprocessed_data, scatter_kws ={'alpha':0.1})

ax.set_title(f'correlation: {pearson_test[0]:.3f} | p_value : {pearson_test[1]:.3f}')

plt.show()

plt.figure(figsize =(8,6))

ax1=sns.violinplot(x='Disease', y='Age', data = preprocessed_data)

ax1.set_title('age distribution across two categoical diseases')

plt.show()

disease_mask = preprocessed_data['Disease'] =='Yes'

disease_ages = preprocessed_data['Age'][disease_mask]

no_disease_ages = preprocessed_data['Age'][~disease_mask]

test_res = ttest_ind(disease_ages, no_disease_ages)

print(f'test for equality of means: statistcs = {test_res[0]},p_value : {test_res[1]}')

#test equality of distributions via kolmogorov-Smirnov test

ks_res = ks_2samp(disease_ages, no_disease_ages)

print(f'ks test for equality of distributions: statistics = {test_res[0]}, p_value = {test_res[1]}')

plt.figure(figsize =(20,8))

sns.violinplot(x= 'Reason for absence', y = 'Age', data = preprocessed_data)

plt.show()

#investigation of impact of education on reason for absence
education_type = ['high_school', 'graduate', 'postgraduate', 'master_phd']

counts = preprocessed_data['Education'].value_counts()

percentages = preprocessed_data['Education'].value_counts(normalize=True)

for edu in education_type:
    print(f'education type ={edu:12s} | counts : {counts[edu]:6.0f} | percentage : {percentages[edu]:4.1f}')

plt.figure(figsize =(8,6))

sns.violinplot(x='Education', y='Absenteeism time in hours', data = preprocessed_data,order=['high_school', 'graduate', 'postgraduate', 'master_phd'])
plt.show()
# compute mean and standard deviation of absence hours
education_types = ["high_school", "graduate", \
                   "postgraduate", "master_phd"]
for educ_type in education_types:
    mask = preprocessed_data["Education"] == educ_type
    hours = preprocessed_data["Absenteeism time in hours"][mask]
    mean = hours.mean()
    stddev = hours.std()
    print(f"Education type: {educ_type:12s} | Mean : {mean:.03f} \
          | Stddev: {stddev:.03f}")

plt.figure(figsize =(10, 16))

sns.countplot(x='Reason for absence', hue='Education', data= preprocessed_data, hue_order = ['high_school','graduate','postgraduate', 'master_phd'])

plt.show()

threshold = 40
total_entries = len(preprocessed_data)
high_school_mask = preprocessed_data['Education'] == 'high_school'
extreme_mask = preprocessed_data['Absenteeism time in hours'] > threshold
prob_high_school = len(preprocessed_data[high_school_mask])/total_entries
prob_graduate = len(preprocessed_data[~high_school_mask])/total_entries
prob_extreme_high_school = len(preprocessed_data[high_school_mask & extreme_mask])/total_entries
prob_extreme_graduate = len(preprocessed_data[~high_school_mask & extreme_mask])/total_entries
cond_prob_extreme_high_school = prob_extreme_high_school/prob_high_school
cond_prob_extreme_graduate = prob_extreme_graduate/prob_graduate
print(f'P(extreme absence | degree=high school)={100*cond_prob_extreme_high_school:3.2f}')
print(f'P(extreme absence | degree != high school) = {100*cond_prob_extreme_graduate:3.2f}')
print(preprocessed_data.head().T)

#plot transportation cost and distance to work against hours

plt.figure(figsize =(10, 6))
sns.jointplot(x ='Distance from Residence to Work', y = 'Absenteeism time in hours', data = preprocessed_data, kind = 'reg')
plt.show()

plt.figure(figsize =(10,6))
sns.jointplot(x='Transportation expense', y = 'Absenteeism time in hours', data = preprocessed_data, kind ='reg')
plt.show()

from scipy.stats import yeojohnson
hours = yeojohnson(preprocessed_data['Absenteeism time in hours'].apply(float))
distances = preprocessed_data['Distance from Residence to Work']
expenses = preprocessed_data['Transportation expense']
plt.figure(figsize =(10,6))
ax= sns.jointplot(x= distances, y= hours[0], kind ='reg')
ax.set_axis_labels('Distance from Residence to Work','Hours_transformed')
plt.show()
ax = sns.jointplot(x=expenses, y=hours[0], kind='reg')
ax.set_axis_labels('Transportation expense', 'Hours_transformed')
plt.show()

distances = preprocessed_data['Distance from Residence to Work']
expenses = preprocessed_data['Transportation expense']
plt.figure(figsize =(10,6))
ax= sns.jointplot(x= distances, y= hours[0], kind ='kde')
ax.set_axis_labels('Distance from Residence to Work','Hours_transformed')
plt.show()
ax = sns.jointplot(x=expenses, y=hours[0], kind='kde')
ax.set_axis_labels('Transportation expense', 'Hours_transformed')
plt.show()
# investigate correlation between the columns
distance_corr = pearsonr(hours[0], distances)
expenses_corr = pearsonr(hours[0], expenses)
print(f"Distances correlation: corr={distance_corr[0]:.3f}, \
pvalue={distance_corr[1]:.3f}")
print(f"Expenses comparison:  corr={expenses_corr[0]:.3f}, \
pvalue={expenses_corr[1]:.3f}")
# count entries per day of the week and month
plt.figure(figsize=(12, 5))
ax = sns.countplot(data=preprocessed_data, \
                   x='Day of the week', \
                   order=["Monday", "Tuesday", \
                          "Wednesday", "Thursday", "Friday"])
ax.set_title("Number of absences per day of the week")
plt.savefig('dow_counts.png', format='png', dpi=300)
plt.figure(figsize=(12, 5))
ax = sns.countplot(data=preprocessed_data, \
                   x='Month of absence', \
                   order=["January", "February", "March", \
                          "April", "May", "June", "July", \
                          "August", "September", "October", \
                          "November", "December", "Unknown"])
ax.set_title("Number of absences per month")
plt.savefig('month_counts.png', format='png', dpi=300)
# analyze average distribution of absence hours 
plt.figure(figsize=(12,5))
sns.violinplot(x="Day of the week", \
               y="Absenteeism time in hours",\
               data=preprocessed_data, \
               order=["Monday", "Tuesday", \
                      "Wednesday", "Thursday", "Friday"])
plt.savefig('exercise_206_dow_hours.png', \
            format='png', dpi=300)
plt.figure(figsize=(12,5))
sns.violinplot(x="Month of absence", \
               y="Absenteeism time in hours",\
               data=preprocessed_data, \
               order=["January", "February", \
                      "March", "April", "May", "June", "July",\
                      "August", "September", "October", \
                      "November", "December", "Unknown"])
plt.savefig('exercise_206_month_hours.png', \
            format='png', dpi=300)
months = ["January", "February", "March", "April", "May", \
          "June", "July", "August", "September", "October", \
          "November", "December"]
for month in months:
    mask = preprocessed_data["Month of absence"] == month
    hours = preprocessed_data["Absenteeism time in hours"][mask]
    mean = hours.mean()
    stddev = hours.std()
    print(f"Month: {month:10s} | Mean : {mean:8.03f} \
| Stddev: {stddev:8.03f}")
thursday_mask = preprocessed_data\
                ["Day of the week"] == "Thursday"
july_mask = preprocessed_data\
            ["Month of absence"] == "July"
thursday_data = preprocessed_data\
                ["Absenteeism time in hours"][thursday_mask]
no_thursday_data = preprocessed_data\
                   ["Absenteeism time in hours"][~thursday_mask]
july_data = preprocessed_data\
            ["Absenteeism time in hours"][july_mask]
no_july_data = preprocessed_data\
               ["Absenteeism time in hours"][~july_mask]
thursday_res = ttest_ind(thursday_data, no_thursday_data)
july_res = ttest_ind(july_data, no_july_data)
print(f"Thursday test result: statistic={thursday_res[0]:.3f}, \
pvalue={thursday_res[1]:.3f}")
print(f"July test result: statistic={july_res[0]:.3f}, \
pvalue={july_res[1]:.3f}")
print(preprocessed_data.head().T)
preprocessed_data["Service time"].hist()











    

