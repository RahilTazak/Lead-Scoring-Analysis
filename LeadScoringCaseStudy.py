#!/usr/bin/env python
# coding: utf-8

# In[523]:


# Importing libraries

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[524]:


# Importing dataset

leads = pd.read_csv('Leads.csv')
leads.head()


# In[525]:


# Checking shape of dataset
leads.shape


# In[526]:


# Checking data types of columns and null values
leads.info()


# In[527]:


# Checking summary of all numerical columns
leads.describe()


# In[528]:


# Checking summary of all columns
leads.describe(include = 'all')


# In[529]:


# Checking for dupicates

leads.duplicated().sum()


# ##  Exploratory Data Analysis

# ### Data Cleaning

# In[530]:


# Checking null values in each column

round(leads.isna().sum()/len(leads.index),2)*100


# In[531]:


# Dropping columns having null values greater than 35%

columns = leads.columns

for i in columns:
    if (round(leads[i].isna().sum()/len(leads.index),2)*100) > 35:
        leads.drop(i, axis = 1, inplace = True)
  


# In[532]:


# Checking null values and shape again

print(round(leads.isna().sum()/len(leads.index),2)*100)
print('-'*50)
print(leads.shape)


# In[533]:


leads.head(3)


# There are no duplicate values in the dataset

# In[534]:


# Dropping unnecessary columns

leads.drop('Prospect ID', axis =1 , inplace = True)
leads.drop('Lead Number', axis =1 , inplace = True)


# In[535]:


leads.head(3)


# In[536]:


# Checking for columns containing 'Select' 

select_cols = leads.isin(["Select"]).any()
print(select_cols[select_cols] == True)
print(sum(select_cols[select_cols] == True))


# In[537]:


# Replacing 'Select' with Null values

leads.replace('Select', np.nan, inplace = True)


# In[538]:


select_cols = leads.isin(["Select"]).any()
print(select_cols[select_cols] == True)
print(sum(select_cols[select_cols] == True))


# In[539]:


# Selecting the columns with the categorical data

cat_cols = leads.select_dtypes(['object', 'category'])
cat_cols.columns.to_list()


# In[540]:


cat_cols.head()


# In[541]:


# Checking value count of each categorical column

for i in cat_cols:
     print('Value count in % for :',i,'\n')
     print(leads[i].value_counts(normalize = True, dropna = False)*100)
     print('-'*50)
     print(' '*50)


# In[542]:


pd.set_option('display.max_columns',None)
cat_cols.describe()


# There are some columns having only 1 unique value. These columns will not provide any significant insight for the model. Hence, These columns should be dropped

# In[543]:


# Dropping columns having single unique value

drop_cols = []
for i in cat_cols.columns:
    if cat_cols[i].nunique() ==1:
        drop_cols.append(i)
leads.drop(drop_cols, axis =1,inplace = True)
leads.describe(include = 'all')


# In[544]:


# Droppig highly imbalance columns

imbalance_cols = ['Do Not Call','Search','Newspaper Article','X Education Forums','Newspaper',
                  'Digital Advertisement', 'Through Recommendations','What matters most to you in choosing a course']
leads = leads.drop(imbalance_cols,axis =1)


# In[545]:


round(leads.isnull().sum()/len(leads.index),2)*100


# In[546]:


# Dropping columns gaving missing values greater than 35%

columns = leads.columns

for i in columns:
    if (round(leads[i].isna().sum()/len(leads.index),2)*100) > 35: 
        leads.drop(i, axis = 1, inplace = True)


# In[547]:


leads.shape


# In[548]:


round(leads.isnull().sum()/len(leads.index),2)*100


# In[549]:


Cols = ['Country','What is your current occupation']

for i in Cols:
    print('Value count in % for : ', i)
    print(leads[i].value_counts(normalize = True, dropna = False))
    print('-'*50)


# Imputing the value in 'Country' column with the mode will make the data highly imbalance. Hence, This column needs to be dropped.

# In[550]:


leads.drop(['Country'], axis =1, inplace = True)


# In[551]:


leads.info()


# In[552]:


leads['Lead Source'].fillna(leads['Lead Source'].mode()[0], inplace = True)


# In[553]:


leads['Last Activity'].fillna(leads['Last Activity'].mode()[0], inplace = True)


# In[554]:


leads['What is your current occupation'].value_counts(dropna = False)


# In[555]:


leads.info()


# In[556]:


leads['What is your current occupation'].fillna(leads['What is your current occupation'].mode()[0], inplace = True)


# In[557]:


leads = leads[~leads['TotalVisits'].isnull()]


# In[558]:


leads.info()


# In[559]:


leads.head()


# In[560]:


#replacing "Lead Source" with low frequency with "Others"

leads['Lead Source'] = leads['Lead Source'].replace(['bing','Click2call','Social Media','Live Chat','youtubechannel',
    'Press_Release','testone','Pay per Click Ads','welearnblog_Home','WeLearn','blog','NC_EDM'],'Others')
leads['Lead Source'] = leads['Lead Source'].replace('google','Google')


# In[561]:


#replacing "Last Activity" with low frequency with "Others"


leads['Last Activity'] = leads['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                               'Had a Phone Conversation', 
                                                               'Approached upfront',
                                                               'View in browser link Clicked',       
                                                               'Email Marked Spam',                  
                                                               'Email Received','Visited Booth in Tradeshow',
                                                               'Resubscribed to emails'],'Others')


# In[562]:


#replacing "Last Notable Activity" with low frequency with "Others"


leads['Last Notable Activity'] = leads['Last Notable Activity'].replace(['Email Bounced',
                                                               'Unreachable','Unsubscribed',
                                                               'Had a Phone Conversation',
                                                               'Form Submitted on Website',
                                                               'Approached upfront',
                                                               'View in browser link Clicked',       
                                                               'Email Marked Spam',                  
                                                               'Email Received',
                                                               'Resubscribed to emails'],'Others')


# ## Handling Outliers

# In[563]:


num_cols = leads.select_dtypes(['int64','float64'])
num_cols = num_cols.drop(['Converted'], axis = 1)


# In[564]:


num_cols.columns.to_list()


# In[565]:



i = 1
for col in num_cols:
    plt.figure(figsize = [18,10])
    plt.subplot(2,2,i)
    sns.boxplot(leads[col])
    i = i + 1
    plt.show()


# In[566]:


num_cols.describe(percentiles = [0.01,0.05,0.25,0.50,0.75,0.95,0.99])


# In[567]:



for col in num_cols:
    q1 = leads[col].quantile(0.01)
    q3 = leads[col].quantile(0.99)
    leads = leads.loc[(leads[col]>=q1) & (leads[col]<=q3)]


# In[568]:



i = 1
for col in num_cols:
    plt.figure(figsize = [18,10])
    plt.subplot(2,2,i)
    sns.boxplot(leads[col])
    i = i + 1
    plt.show()


# In[569]:


leads.head()


# ###  Univariate and Bivariate Analysis

# In[570]:


plt.figure(figsize=[14,8])
plt.title('Converted vs Non-converted Leads')
leads.Converted.value_counts(normalize = True).plot.pie(autopct = "%1.0f%%")
plt.show()


# 38% people have converted to leads while 62% didn't convert

# In[571]:


cat_cols = leads.select_dtypes(['object', 'category'])
cat_cols.columns.to_list()


# In[572]:


# Plotting graph for categorical columns

i = 1
for col in cat_cols:
    plt.figure(figsize=(30,50))
    plt.subplot(14,2,i)
    sns.countplot(x = col, hue = leads.Converted, data = leads)
    plt.xticks(rotation = 90)
    i = i+1
plt.show()


# Majority of customers have a lead source of Google and Direct Traffic. Lead source from Google has a highest probability of conversion followed by Direct Traffic

# Customers whose last activity was SMS Sent are more likely to be converted followed by the customers having last activity was Email Opened.

# Maximum number of converted leads have occupation defined as Unemployed.

# In[573]:


leads.head(2)


# In[574]:


leads['Do Not Email'] = leads['Do Not Email'].map({'Yes':1, 'No':0})
leads['A free copy of Mastering The Interview'] = leads['A free copy of Mastering The Interview'].map({'Yes':1, 'No':0})


# In[575]:


leads.head(2)


# In[576]:


# Numerical data analysis

plt.figure(figsize=(18, 6))
sns.pairplot(data=leads,vars=num_cols,hue="Converted")                                  
plt.show()


# In[577]:


# Heatmap for numerical columns

num_col = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']
sns.heatmap(leads[num_col].corr(), cmap = 'Blues', annot = True)
plt.show()


# In[578]:


plt.figure(figsize=(16, 10))
plt.subplot(2,2,1)
plt.title('Total visits vs Converted')
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = leads)
plt.subplot(2,2,2)
plt.title('Page Views Per Visit')
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = leads)
plt.subplot(2,2,3)
plt.title('Total Time Spent on Website')
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = leads)
plt.show()


# Leads spending more time on website are more likely to be converted

# In[579]:


for i in leads.columns:
     print('Value count in % for :',i,'\n')
     print((leads[i].value_counts(normalize = True, dropna = False)*100))
     print('-'*50)
     print(' '*50)


# In[580]:


leads.info()


# # Data Preparation

# ### Dummy Variables

# In[581]:


cols = [i for i in leads.select_dtypes(['object']).columns if len(leads[i].value_counts()) > 2]
cols


# In[582]:


dummy_cols = pd.get_dummies(leads[cols], drop_first = True)
dummy_cols.head()


# In[583]:


leads = pd.concat([leads, dummy_cols], axis = 1)
leads.drop(cols, axis=1, inplace = True)


# In[584]:


leads.head()


# ### Test-Train Split

# In[585]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[586]:


# Putting predictor variables to X
X = leads.drop('Converted', axis=1)

# Putting Target variables to y
y = leads["Converted"]


# In[587]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, train_size = 0.7, random_state = 100)


# ### Feature Scaling and Model Building

# In[588]:


scaler = StandardScaler()
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_train.head()                                                                                                             


# ### Model 1

# In[589]:


logreg = LogisticRegression()
rfe = RFE(logreg,15)
rfe = rfe.fit(X_train,y_train)


# In[590]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[591]:


col = X_train.columns[rfe.support_]
X_train_sm = sm.add_constant(X_train[col])
m1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial()).fit()
m1.summary()


# ### Model 2

# In[592]:


col = col.drop('What is your current occupation_Housewife',1)
X_train_sm2 = sm.add_constant(X_train[col])
m2 = sm.GLM(y_train,X_train_sm2, family = sm.families.Binomial()).fit()
m2.summary()


# In[593]:


vif = pd.DataFrame()
X = X_train[col]
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif.sort_values(by = 'VIF',ascending = False)


# ### Model 3

# In[594]:


col = col.drop('Lead Source_Reference',1)
X_train_sm3 = sm.add_constant(X_train[col])
m3 = sm.GLM(y_train,X_train_sm3, family = sm.families.Binomial()).fit()
m3.summary()


# In[595]:


vif = pd.DataFrame()
X = X_train[col]
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif.sort_values(by = 'VIF',ascending = False)


# ### Model 4

# In[596]:


col = col.drop('What is your current occupation_Working Professional',1)
X_train_sm4 = sm.add_constant(X_train[col])
m4 = sm.GLM(y_train,X_train_sm4, family = sm.families.Binomial()).fit()
m4.summary()


# In[597]:


vif = pd.DataFrame()
X = X_train[col]
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif.sort_values(by = 'VIF',ascending = False)


# ### Model Evaluation

# #### Making Predictions on Train Set

# In[598]:


y_train_pred = m4.predict(X_train_sm4)
y_train_pred


# In[599]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred


# In[600]:


# Creating a dataframe

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Predicted'] = y_train_pred_final['Converted_prob'].map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[601]:


from sklearn import metrics

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[602]:


print(metrics.accuracy_score(y_train_pred_final['Converted'],y_train_pred_final['Predicted']))


# In[603]:


TP = confusion[1,1]   # True Positive
TN = confusion[0,0]   # True Negative
FP = confusion[0,1]   # False Positive
FN = confusion[1,0]   # False Negative


# In[604]:


# Sensitivity

TP/(float(TP + FN))


# In[605]:


# Specificity

TN/(float(TN + FP))


# In[606]:


# False Poitive Rate

FP/float(FP + TN)


# In[607]:


# Positive Predictive Value

TP/float(TP+FP)


# In[608]:


# Negative Predictive Value

TN/float(TN+FN)


# In[609]:


# Plotting ROC Curve

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[610]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# ##### Finding optimal cut-off

# In[611]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[612]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[613]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.vlines(x=0.35,ymax = 1.0,ymin=0.0,color='r',linestyle='--')

plt.show()


# 0.35 is the optimal cut-off point

# In[614]:


y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()


# In[615]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','final_Predicted','Lead_Score']].head()


# In[616]:


# Accuracy

round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted),2)


# In[617]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[618]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[619]:


# Sensitivity

round(TP / float(TP+FN),2)


# In[620]:


# Specificity

round(TN / float(TN+FP),2)


# In[621]:


# Precision

TP / (TP + FP)
round(confusion[1,1]/(confusion[0,1]+confusion[1,1]),2)


# In[622]:


# Recall

TP / TP + FN

round(confusion[1,1]/(confusion[1,0]+confusion[1,1]),2)


# In[623]:


from sklearn.metrics import precision_score, recall_score


# In[624]:


round(precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted),2)


# In[625]:


round(recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted),2)


# In[626]:


from sklearn.metrics import precision_recall_curve


# In[627]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[628]:


plt.plot(thresholds, p[:-1], "g-",label = 'Precision')
plt.plot(thresholds, r[:-1], "r-", label = 'Recall')
plt.vlines(x=0.42,ymax = 1.0,ymin=0.0,color='r',linestyle='--')
plt.legend(loc = 'lower left')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.show()


# ### Making Predictions on Test Data

# In[629]:


num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[630]:


X_test = X_test[col]
X_test.head()


# In[631]:


X_test_sm = sm.add_constant(X_test)


# In[632]:


y_test_pred = m4.predict(X_test_sm)
y_test_pred.head()


# In[633]:


y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[634]:


y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[635]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[636]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[637]:


y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
y_pred_final.head()


# In[638]:



y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))
y_pred_final.head()


# In[639]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final.head()


# In[640]:


# Accuracy

round(metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted),2)


# In[641]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[642]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[643]:


# Sensitivity

round(TP / float(TP+FN),2)


# In[644]:


# Specificity

round(TN / float(TN+FP),2)


# In[645]:


round(precision_score(y_pred_final.Converted , y_pred_final.final_Predicted),2)


# In[646]:


round(recall_score(y_pred_final.Converted, y_pred_final.final_Predicted),2)


# ## Final Observation:
# 
# ### Train Data :
#        Accuracy    : 81%
#        Sensitivity : 81%
#        Specificity : 81%          
# 
# ### Test Data :
#        Accuracy    : 80%
#        Sensitivity : 83%
#        Specificity : 78%    
# 

# In[ ]:




