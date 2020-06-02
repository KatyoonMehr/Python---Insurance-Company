# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:11:23 2020

@author: Katayoon Sadeghi mehr
"""


import numpy as np
import pandas as pd
import os

os.chdir('C:\\Users\\Rayan\\Desktop\\Kati\\Data Science\\Python Project\\My Project')
os.getcwd()


covers = pd.read_csv('cover_test_dataset.csv', 
                     parse_dates=['installation_at', 'request_created_at',
                    'agent_responded_to_customer_at','customer_sold_at'])

covers.head()

covers.columns
covers.info()


covers.isnull().any()
covers.isnull().sum()


covers["installation_at"] = pd.to_datetime(covers["installation_at"])
covers["request_created_at"] = pd.to_datetime(covers["request_created_at"])
covers["agent_responded_to_customer_at"] = pd.to_datetime(covers["agent_responded_to_customer_at"])
covers["customer_sold_at"] = pd.to_datetime(covers["customer_sold_at"])



## Dealing with missing values and frequencies
covers_n = covers.fillna(value={'customer_sold_at': "Not Sold", 
                                    'installation_at': "Not Install", 
                                    'request_created_at': "No Quote",
                                    'agent_responded_to_customer_at': "No Contact"},  
                                     axis=0, inplace= False)

covers_n.loc[covers_n['installation_at'] == "Not Install", 'install'] = 0
covers_n.loc[covers_n['installation_at'] != "Not Install", 'install'] = 1
covers_n['install'].value_counts()

covers_n.loc[covers_n['request_created_at'] == "No Quote", 'quote'] = 0
covers_n.loc[covers_n['request_created_at'] != "No Quote", 'quote'] = 1
covers_n['quote'].value_counts()

covers_n.loc[covers_n['agent_responded_to_customer_at'] == "No Contact", 'contact'] = 0
covers_n.loc[covers_n['agent_responded_to_customer_at'] != "No Contact", 'contact'] = 1
covers_n['contact'].value_counts()

covers_n.loc[covers_n['customer_sold_at'] == "Not Sold", 'sold'] = 0
covers_n.loc[covers_n['customer_sold_at'] != "Not Sold", 'sold'] = 1
covers_n['sold'].value_counts()

pd.crosstab(covers_n.install, covers_n.sold)
pd.crosstab(covers_n.quote, covers_n.sold)
pd.crosstab(covers_n.contact, covers_n.sold)



from datetime import datetime, date


type(covers.installation_at)
type(covers.request_created_at)
type(covers.agent_responded_to_customer_at)
type(covers.customer_sold_at)


covers['year_install'] = covers['installation_at'].dt.year
covers['quarter_install'] = covers['installation_at'].dt.quarter
covers['month_install'] = covers['installation_at'].dt.month
covers['day_install'] = covers['installation_at'].dt.day
covers['weekday_install'] = covers['installation_at'].dt.weekday_name
covers['nday_install'] = covers['installation_at'].dt.dayofweek
covers['hour_install'] = covers['installation_at'].dt.hour
covers['minute_install'] = covers['installation_at'].dt.minute

covers['year_quote'] = covers['request_created_at'].dt.year
covers['quarter_quote'] = covers['request_created_at'].dt.quarter
covers['month_quote'] = covers['request_created_at'].dt.month
covers['day_quote'] = covers['request_created_at'].dt.day
covers['weekday_quote'] = covers['request_created_at'].dt.weekday_name
covers['nday_quote'] = covers['request_created_at'].dt.dayofweek
covers['hour_quote'] = covers['request_created_at'].dt.hour
covers['minute_quote'] = covers['request_created_at'].dt.minute

covers['year_contact'] = covers['agent_responded_to_customer_at'].dt.year
covers['quarter_contact'] = covers['agent_responded_to_customer_at'].dt.quarter
covers['month_contact'] = covers['agent_responded_to_customer_at'].dt.month
covers['day_contact'] = covers['agent_responded_to_customer_at'].dt.day
covers['weekday_contact'] = covers['agent_responded_to_customer_at'].dt.weekday_name
covers['nday_contact'] = covers['agent_responded_to_customer_at'].dt.dayofweek
covers['hour_contact'] = covers['agent_responded_to_customer_at'].dt.hour
covers['minute_contact'] = covers['agent_responded_to_customer_at'].dt.minute

covers['year_sold'] = covers['customer_sold_at'].dt.year
covers['quarter_sold'] = covers['customer_sold_at'].dt.quarter
covers['month_sold'] = covers['customer_sold_at'].dt.month
covers['day_sold'] = covers['customer_sold_at'].dt.day
covers['weekday_sold'] = covers['customer_sold_at'].dt.weekday_name
covers['nday_sold'] = covers['customer_sold_at'].dt.dayofweek
covers['hour_sold'] = covers['customer_sold_at'].dt.hour
covers['minute_sold'] = covers['customer_sold_at'].dt.minute



covers['sold1'] = covers['customer_sold_at']
covers.fillna(value={'sold1': "Not Sold"}, axis=0, inplace= True)
covers.loc[covers['sold1'] == "Not Sold", 'sold'] = 0
covers.loc[covers['sold1'] != "Not Sold", 'sold'] = 1
covers['sold'].value_counts()
del covers['sold1']


covers.to_csv(r'C:\\Users\\Rayan\\Desktop\\Kati\\Covers\coverscodes.csv')


import matplotlib.pylab as plt
import seaborn as sns

plt.hist(covers.month_install, bins=12, density=True, color='blue')
plt.xlabel('Month')
plt.ylabel('percentage')
plt.title('Month in which the application was installed', fontsize=15, color='blue')

plt.hist(covers.month_quote, bins=12, density=True, color='red')
plt.xlabel('Month')
plt.ylabel('percentage')
plt.title('Month in which the quote was done', fontsize=15, color='red')

plt.hist(covers.month_contact, bins=12, density=True, color='orange')
plt.xlabel('Month')
plt.ylabel('percentage')
plt.title('Month in which the agent contact the potential customer', fontsize=15, color='orange')

plt.hist(covers.month_sold, bins=12, density=True, color='green')
plt.xlabel('Month')
plt.ylabel('percentage')
plt.title('Month in which Insurance was sold', fontsize=15, color='green')


plt.hist(covers.hour_install, bins=24, density=True, color='blue')
plt.xlabel('Hour')
plt.ylabel('percentage')
plt.title('Hours at which the application was installed', fontsize=15, color='blue')

plt.hist(covers.hour_quote, bins=24, density=True, color='red')
plt.xlabel('Hour')
plt.ylabel('percentage')
plt.title('Hours at which the quote was done', fontsize=15, color='red')

plt.hist(covers.hour_contact, bins=24, density=True, color='orange')
plt.xlabel('Hour')
plt.ylabel('percentage')
plt.title('Hours at which the agent contacted the potential customer', fontsize=15, color='orange')

plt.hist(covers.hour_sold, bins=24, density=True, color='green')
plt.xlabel('Hour')
plt.ylabel('percentage')
plt.title('Hours at which the insurance was sold', fontsize=15, color='green')

count_month_install = covers.month_install.value_counts()
count_month_quote = covers.month_quote.value_counts()
count_month_contact = covers.month_contact.value_counts()
count_month_sold = covers.month_sold.value_counts()

count_weekday_install = covers.weekday_install.value_counts()
count_weekday_quote = covers.weekday_quote.value_counts()
count_weekday_contact = covers.weekday_contact.value_counts()
count_weekday_sold = covers.weekday_sold.value_counts()


d1 = covers['customer_sold_at'] - covers['installation_at']
d1.dropna(axis=0, inplace= True) 
d1.drop([61217, 56933, 58427, 89468, 25820], inplace = True)
d1.describe()

d2 = covers['customer_sold_at'] - covers['request_created_at']
d2.dropna(axis=0, inplace= True) 
d2.drop([61217, 56933, 58427, 89468, 25820, 444, 48188], inplace = True)
d2.describe()

d3 = covers['customer_sold_at'] - covers['agent_responded_to_customer_at']
d3.dropna(axis=0, inplace= True) 
d3.drop([61217, 56933, 48339, 50533, 48369, 89468, 52991, 51505, 58427,\
         25820, 53432, 47982, 63699, 83903, 73209, 47830, 53965, 45864, \
         72224, 107130, 75506], inplace = True)
d3.describe()


## Sales conversion based on season 
summer = covers[(covers.month_install == 7) | (covers.month_install == 8) | (covers.month_install == 9)]
summer.isnull().sum()

winter = covers[(covers.month_install == 1) | (covers.month_install == 2) | (covers.month_install == 3)]
winter.isnull().sum()







###########################################################################
## For the new dataset of customers


dem = pd.read_csv('Dem.csv', delimiter=',')
insold = pd.read_csv('covers_sold.csv', delimiter=',')
dem.describe()
insold.describe()

#Merge the datasets
fulldata = pd.merge(dem, insold, on=['user_id'])
fulldata.head()
fulldata.columns


# Removes rows with negtive values for premium
fulldata = fulldata.loc[fulldata.Premium > 0]




#Get the first two characters of INS_id to know which type of Insurance
fulldata['ins_type'] = fulldata['INS_id'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories:
fulldata['ins_type'] = fulldata['ins_type'].map({'AU':'Auto',
                                                 'HM':'Home',
                                                 'HT':'Health',
                                                 'LF':'Life'})
fulldata['ins_type'].value_counts()

#Barchart of Insurance type
fulldata['ins_type'].value_counts().plot(kind = 'bar', color='red', fontsize=10)
plt.title('Frequency of Insurance type')



## City 
fulldata['City'].value_counts()

mean_premium_city=fulldata.groupby(['City'])['Premium'].agg(np.mean)
mean_premium_city

max_premium_city=fulldata.groupby(['City'])['Premium'].agg(np.max)
max_premium_city

min_premium_city=fulldata.groupby(['City'])['Premium'].agg(np.min)
min_premium_city


# Barchart of the numver of applicants per cities
city_count = fulldata['City'].value_counts()
plt.figure(figsize=(10,5))
chart = sns.barplot(city_count.index, city_count.values) 
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title('Histogram of Customers per Cities')
plt.ylabel('Frequency')
plt.xlabel('City', fontsize=0.1)


# Gender
fulldata['Gender'].value_counts()

slices_genders = [3811, 1921]
labels = ['Male', 'Female']
colors = ['blue', 'red']
plt.pie(slices_genders, labels=labels, colors=colors)
plt.title("Pie chart gender of customers")
plt.show()



# Boxplot of the Premium by type of the insurance and gender
ax = sns.catplot(x="ins_type", y="Premium", hue="Gender", kind="box", data=fulldata, palette="Set1")
ax.fig.suptitle("Box plot of Primum based on Type of insurance and Gender")

ax = sns.catplot(x="Gender", y="Premium", kind="box", data=fulldata, palette="Set1")
ax.fig.suptitle("Box plot of Primum based on Gender")

mean_premium_gender=fulldata.groupby(['Gender'])['Premium'].agg(np.mean)
mean_premium_gender

max_premium_gender=fulldata.groupby(['Gender'])['Premium'].agg(np.max)
max_premium_gender

min_premium_gender=fulldata.groupby(['Gender'])['Premium'].agg(np.min)
min_premium_gender


# For Auto Insurance
fulldata_AU = fulldata.loc[fulldata.ins_type == 'Auto']
Gen_Acc = pd.crosstab(fulldata_AU.Gender, fulldata_AU.Accident, margins=True, margins_name="Total")
Gen_Tic = pd.crosstab(fulldata_AU.Gender, fulldata_AU.Ticket, margins=True, margins_name="Total")
Gen_ins_cal = pd.crosstab(fulldata_AU.Gender, fulldata_AU.ins_Cancelation, margins=True, margins_name="Total")


mean_premiumAU_gender=fulldata_AU.groupby(['Gender'])['Premium'].agg(np.mean)
mean_premiumAU_gender

max_premiumAU_gender=fulldata_AU.groupby(['Gender'])['Premium'].agg(np.max)
max_premiumAU_gender

min_premiumAU_gender=fulldata_AU.groupby(['Gender'])['Premium'].agg(np.min)
min_premiumAU_gender


np.mean(fulldata_AU['Premium'])


#Encoding Gender

fulldata['Gender_C'] = fulldata.Gender.map({'M':1, 
                                            'F':2})

fulldata['City_C'] = fulldata.City.map({'Boston':1,
                                        'Chicago':2,
                                        'Dallas':3,
                                        'Houston':4,
                                        'Los Angeles':5,
                                        'Miami':6,
                                        'New york':7,
                                        'Philadelphia':8,
                                        'Phoenix':9,
                                        'San Antonio':10,
                                        'San Diego':11,
                                        'San Fransisco':12,
                                        'Washington DC':13})

fulldata['nday_install'] = fulldata.weekday_install.map({'Monday':1,
                                                         'Tuesday':2,
                                                         'Wednesday':3,
                                                         'Thursday':4,
                                                         'Friday':5,
                                                         'Saturday':6,
                                                         'Sunday':7})

fulldata['nday_quote'] = fulldata.weekday_quote.map({'Monday':1,
                                                     'Tuesday':2,
                                                     'Wednesday':3,
                                                     'Thursday':4,
                                                     'Friday':5,
                                                     'Saturday':6,
                                                     'Sunday':7})

fulldata['nday_contact'] = fulldata.weekday_contact.map({'Monday':1,
                                                         'Tuesday':2,
                                                         'Wednesday':3,
                                                         'Thursday':4,
                                                         'Friday':5,
                                                         'Saturday':6,
                                                         'Sunday':7})

fulldata['nday_sold'] = fulldata.weekday_sold.map({'Monday':1,
                                                   'Tuesday':2,
                                                   'Wednesday':3,
                                                   'Thursday':4,
                                                   'Friday':5,
                                                   'Saturday':6,
                                                   'Sunday':7})

## Correlation
fulldata['Age'].corr(fulldata['Premium'])






fulldata_AU['ins_Cancelation'].value_counts()
fulldata_AU['Ticket'].value_counts()
fulldata_AU['Accident'].value_counts()



''' ANOVA '''

from statsmodels.formula.api import ols
import statsmodels.api as sm

ANOVA_Model = ols('Premium ~ C(ins_Cancelation)+C(Ticket)+(Accident)', data=fulldata).fit()
table = sm.stats.anova_lm(ANOVA_Model, typ=2) # Type 2 ANOVA DataFrame
print(table)



''' Linear Regression '''

fulldata_NMiss = fulldata.dropna(axis=0, subset=None, inplace= False) 

y = fulldata_NMiss['Premium']
X = fulldata_NMiss[['City_C', 'Gender_C', 'Age', 'ins_Cancelation', 'Ticket', 'Accident', \
             'month_install', 'day_install', 'nday_install', 'hour_install', \
             'month_quote', 'day_quote', 'nday_quote', 'hour_quote', \
             'month_contact', 'day_contact', 'nday_contact', 'hour_contact', \
             'month_sold', 'day_sold', 'nday_sold', 'hour_sold']]

          
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
sc = StandardScaler()
X_train_S=sc.fit_transform(X_train)
X_test_S=sc.transform(X_test)

model_lr= LinearRegression()
model_lr.fit(X_train_S, y_train)

y_pred_lr = model_lr.predict(X_test_S)

print(model_lr.intercept_)
print(model_lr.coef_)



ols = sm.OLS(y_train, X_train_S)
lr=ols.fit()
pvalues = lr.pvalues
print(lr.summary())



yF = fulldata_NMiss['Premium']
XF = fulldata_NMiss[['ins_Cancelation', 'Ticket', 'Accident']]

XF_train, XF_test, yF_train, yF_test = train_test_split(XF, yF, test_size=0.25, random_state=100)
sc = StandardScaler()
XF_train_S=sc.fit_transform(XF_train)
XF_test_S=sc.transform(XF_test)

model_lr= LinearRegression()
model_lr.fit(XF_train_S, yF_train)

y_pred_lrF = model_lr.predict(XF_test_S)

print(model_lr.intercept_)
print(model_lr.coef_)

ols = sm.OLS(yF_train, XF_train_S)
lr=ols.fit()
pvalues = lr.pvalues
print(lr.summary())


R2_trainF=model_lr.score(XF_train, yF_train)
R2_testF=model_lr.score(XF_test, yF_test)



''' Adaboost ''' 

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(n_estimators=10, random_state=10)
ada.fit(X_train_S, y_train)
y_pred_ada = ada.predict(X_test_S)
ada.score(X_test_S, y_test)


scores=[]
for i in [2,3,4,5,10, 20]:
    ada = AdaBoostRegressor(n_estimators=i, random_state=10)
    ada.fit(X_train_S, y_train)
    scores.append(ada.score(X_test_S, y_test))  
    
plt.plot([2,3,4,5,10, 20], scores)




''' Random Forest '''

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=95, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf.score(X_test, y_test)
  
scores=[]
for i in [10, 100, 1000, 2500]:
    rf = RandomForestRegressor(n_estimators=i, random_state=10)
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))   
    
plt.plot([10, 100, 1000, 2500], scores)
plt.xlabel("no of modelss")
plt.ylabel("Accuracy on the test dataset")




''' SVR '''

from sklearn.svm import SVR
                         
svr = SVR(kernel = 'poly', gamma = 5 , C=1)
svr.fit(X_train_S, y_train)
y_pred_svr = svr.predict(X_test_S)
svr.score(X_test_S, y_test)


from sklearn.model_selection import GridSearchCV
param_dict = {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma' : [0.1, 1, 5],
                'C': [1, 10]                
            }

grid = GridSearchCV(SVR(), param_dict, cv=4)
grid.fit(X_train_S, y_train)

grid.best_score_

grid.best_params_




''' KNN '''

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, \
                         precision_score, confusion_matrix, classification_report
  

                      
model_knn = KNeighborsRegressor(n_neighbors=30)

model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
classifier.score(X_test, y_test)

score=[]
for i in range (5, 100):
    classifier = KNeighborsRegressor(n_neighbors = i, weights='uniform')
    classifier.fit(X_train, y_train)
    sc=classifier.score(X_test, y_test)
    score.append(sc)

plt.plot(range(5,100), score)
