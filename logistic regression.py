#!/usr/bin/env python
# coding: utf-8

# # LEAD SCORING CASE STUDY

# # Importing libraries and loading data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings(action = 'ignore')


# In[2]:


lead_df = pd.read_csv("Leads.csv")


# # Understanding data

# In[3]:


lead_df.head()


# In[4]:


lead_df.shape


# In[5]:


lead_df.info()


# In[6]:


lead_df.describe()


# # Data Preperation

# # DEALING WITH MISSING VALUES

# In[7]:


lead_df.isnull().mean()


# In[8]:


median =["Asymmetrique Activity Score", "Asymmetrique Profile Score",]


# In[9]:


mode = ["Country", "Specialization","How did you hear about X Education",
         "What is your current occupation",
"What matters most to you in choosing a course", "Tags","Lead Quality", "Lead Profile", "City", "Asymmetrique Activity Index",
"Asymmetrique Profile Index"]


# # DROPPING ALL THE COLOUMNS WITH MISSING VALUES LESS THAN 5%

# In[10]:


lead_df = lead_df.drop(["Page Views Per Visit"],axis=1)


# In[11]:


lead_df = lead_df.drop(["TotalVisits"],axis=1)


# In[12]:


lead_df = lead_df.drop(["Last Activity"],axis=1)


# # IMPUTING THE MISSING VALUES OF COLUMNS WITH CONTINOUS VARIABLES WITH THEIR RESPECTIVE MEDIAN VALUES

# In[13]:


med_1 = lead_df["Asymmetrique Activity Score"].median()
lead_df["Asymmetrique Activity Score"] = lead_df["Asymmetrique Activity Score"].fillna(med_1)


# In[14]:


med_2 = lead_df["Asymmetrique Profile Score"].median()
lead_df["Asymmetrique Profile Score"] = lead_df["Asymmetrique Profile Score"].fillna(med_2)


# # IMPUTING THE MISSING VALUES OF COLUMNS WITH CONTINOUS VARIABLES WITH THEIR RESPECTIVE MODE VALUES

# In[15]:


mod_1 = lead_df["How did you hear about X Education"].mode()
lead_df["How did you hear about X Education"] = lead_df["How did you hear about X Education"].fillna(mod_1)


# In[16]:


mod_2 =  lead_df["Country"].mode()
lead_df["Country"] = lead_df["Country"].fillna(mod_2)


# In[17]:


mod_3 =  lead_df["Specialization"].mode()
lead_df["Specialization"] = lead_df["Specialization"].fillna(mod_3)


# In[18]:


mod_4 =  lead_df["What is your current occupation"].mode()
lead_df["What is your current occupation"] = lead_df["What is your current occupation"].fillna(mod_4)


# In[19]:


mod_5 =  lead_df["Tags"].mode()
lead_df["Tags"] = lead_df["Tags"].fillna(mod_5)


# In[20]:


mod_6 =  lead_df["Lead Quality"].mode()
lead_df["Lead Quality"] = lead_df["Lead Quality"].fillna(mod_6)


# In[21]:


mod_7 =  lead_df["City"].mode()
lead_df["City"] = lead_df["City"].fillna(mod_7)


# In[22]:


mod_8 =  lead_df["Asymmetrique Activity Index"].mode()
lead_df["Asymmetrique Activity Index"] = lead_df["Asymmetrique Activity Index"].fillna(mod_8)


# In[23]:


mod_9 =  lead_df["Asymmetrique Profile Index"].mode()
lead_df["Asymmetrique Profile Index"] = lead_df["Asymmetrique Profile Index"].fillna(mod_9)


# In[24]:


lead_df.isnull().mean()


# # DEALING WITH OUTLIERS 

# In[25]:


sns.boxplot(lead_df["Asymmetrique Activity Score"])
plt.show()


# In[26]:



sns.boxplot(lead_df["Asymmetrique Profile Score"])
plt.show()


# In[27]:


sns.boxplot(lead_df["Lead Number"])
plt.show()


# In[28]:



sns.boxplot(lead_df["Converted"])
plt.show()


# In[29]:


sns.boxplot(lead_df["Total Time Spent on Website"])
plt.show()


# # No outliers detected

# # UNIVARIATE ANALYSIS

# cont_vars = ["Lead Number","Converted","TotalVisits","Total Time Spent on Website","Page Views Per Visit",
#              "Asymmetrique Activity Score","Asymmetrique Profile Score"]
# 
# cat_vars = ["Lead Origin", "Lead Source" , "Do Not Email" , "Do Not Call" , "Last Activity" ,  "Specialization",
#             "How did you hear about X Education", "What is your current occupation","What matters most to you in choosing a course",
#             "Tags","Lead Quality", "Lead Profile", "City", "Asymmetrique Activity Index","Asymmetrique Profile Index" ,
#             "Search" , "Newspaper Article" , "X Education Forums" , "Newspaper","Digital Advertisement" ,
#             "Through Recommendations" , "Update me on Supply Chain Content" , "Get updates on DM Content" ,
#             "I agree to pay the amount through cheque" , "A free copy of Mastering The Interview,Last Notable Activity"]  
#             
#             

# In[30]:


sns.distplot(lead_df["Lead Number"])
plt.show()


# In[31]:


sns.countplot(lead_df["Converted"])
plt.show()


# Only 30% of leads are converted

# In[32]:


sns.distplot(lead_df["Total Time Spent on Website"])
plt.show()


# Most of the lead have spent 1000 - 1500 units on website

# In[33]:


sns.countplot(lead_df["Asymmetrique Activity Score"])
plt.show()


# In[34]:


sns.countplot(lead_df["Asymmetrique Profile Score"])
plt.show()


# In[35]:


plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df.Country, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Majority of leads are from India

# In[36]:


plt.figure(figsize=(10,5))
s1=sns.countplot(lead_df.City, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Majority of leads are from Mumbai

# In[37]:



plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df.Specialization, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Majority of leads are from Finance, Marketing and HR respectively.

# In[38]:


s1=sns.countplot(lead_df['What is your current occupation'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Majority are unemployed.

# In[39]:


s1=sns.countplot(lead_df['What matters most to you in choosing a course'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# 'Better Career Prospects' is what attracts leads.

# In[40]:


plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df['Tags'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[41]:


plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df['Lead Source'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Majority of lead are sourced from Google.

# In[42]:


plt.figure(figsize=(8,5))
s1=sns.countplot(lead_df['Lead Origin'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# # Bivariate Analysis

# In[43]:


plt.figure(figsize=[16,15])
sns.barplot(lead_df["Lead Origin"],lead_df["Converted"],)
plt.show()


# In[44]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Lead Source"],lead_df["Converted"],)
plt.show()


# In[45]:


plt.figure(figsize=[16,15])
sns.barplot(lead_df["Do Not Email"],lead_df["Converted"],)
plt.show()


# In[46]:


plt.figure(figsize=[16,15])
sns.barplot(lead_df["Do Not Call"],lead_df["Converted"],)
plt.show()


# In[47]:


plt.figure(figsize=[40,55])
sns.barplot(lead_df["Specialization"],lead_df["Converted"],)
plt.show()


# In[48]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["How did you hear about X Education"],lead_df["Converted"],)
plt.show()


# In[49]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["What is your current occupation"],lead_df["Converted"],)
plt.show()


# In[50]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["What matters most to you in choosing a course"],lead_df["Converted"],)
plt.show()


# In[51]:


plt.figure(figsize=[40,45])
sns.barplot(lead_df["Tags"],lead_df["Converted"],)
plt.show()


# In[52]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Lead Quality"],lead_df["Converted"],)
plt.show()


# In[53]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Lead Profile"],lead_df["Converted"],)
plt.show()


# In[54]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["City"],lead_df["Converted"],)
plt.show()


# In[55]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Asymmetrique Activity Index"],lead_df["Converted"],)
plt.show()


# In[56]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Asymmetrique Profile Index"],lead_df["Converted"],)
plt.show()


# In[57]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Search"],lead_df["Converted"],)
plt.show()


# In[58]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Newspaper Article"],lead_df["Converted"],)
plt.show()


# In[59]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["X Education Forums"],lead_df["Converted"],)
plt.show()


# In[60]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Newspaper"],lead_df["Converted"],)
plt.show()


# In[61]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Digital Advertisement"],lead_df["Converted"],)
plt.show()


# In[62]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Through Recommendations"],lead_df["Converted"],)
plt.show()


# In[63]:



sns.barplot(lead_df["Update me on Supply Chain Content"],lead_df["Converted"],)
plt.show()


# In[64]:



sns.barplot(lead_df["Get updates on DM Content"],lead_df["Converted"],)
plt.show()


# In[65]:


sns.barplot(lead_df["I agree to pay the amount through cheque"],lead_df["Converted"],)
plt.show()


# In[66]:


sns.barplot(lead_df["A free copy of Mastering The Interview"],lead_df["Converted"],)
plt.show()


# In[67]:


plt.figure(figsize=[30,35])
sns.barplot(lead_df["Last Notable Activity"],lead_df["Converted"],)
plt.show()


# # Multivariate Analysis

# In[68]:


sns.pairplot(lead_df)


# In[183]:



sns.heatmap(lead_df.corr(),annot = True)
plt.show()


# # DUMMY VARIABLE CREATION

# In[70]:


cat_cols= lead_df.select_dtypes(include=['object']).columns
cat_cols


# In[71]:


varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
lead_df[varlist] = lead_df[varlist].apply(binary_map)


# In[72]:


dummy = pd.get_dummies(lead_df[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)

lead_df = pd.concat([lead_df,dummy],1)


# In[73]:


lead_df['Specialization'] = lead_df['Specialization'].replace(np.nan, 'Not Specified')


# In[74]:


dummy = pd.get_dummies(lead_df['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[75]:


lead_df['Lead Source'] = lead_df['Lead Source'].replace(np.nan,'Others')
lead_df['Lead Source'] = lead_df['Lead Source'].replace('google','Google')
lead_df['Lead Source'] = lead_df['Lead Source'].replace('Facebook','Social Media')
lead_df['Lead Source'] = lead_df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')    


# In[76]:


dummy = pd.get_dummies(lead_df['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[77]:


lead_df['Last Notable Activity'] = lead_df['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')


# In[78]:


dummy = pd.get_dummies(lead_df['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[79]:


lead_df['Tags'] = lead_df['Tags'].replace(np.nan,'Not Specified')


# In[80]:


dummy = pd.get_dummies(lead_df['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[81]:


lead_df.drop(cat_cols,1,inplace = True)


# In[82]:


lead_df.head()


# # MODEL BUILDING

# # SPLITTING DATA FOR TRAIN AND TEST

# In[83]:


from sklearn.model_selection import train_test_split

# Putting response variable to y
y = lead_df['Converted']

y.head()

X=lead_df.drop('Converted', axis=1)


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[85]:


X_train.info()


# # SCALING

# In[86]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# In[95]:


from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)


# In[96]:


y_pred_test = logreg.predict(X_test)

y_pred_test


# In[102]:


list(zip(X_train.columns))


# In[104]:


col = X_train.columns
col


# In[105]:


X_train.columns


# In[106]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[107]:


#dropping column with high p-value

col = col.drop('Lead Source_Referral Sites',1)


# In[108]:


#BUILDING MODEL #2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[109]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[110]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[111]:


#dropping variable with high VIF

col = col.drop('Last Notable Activity_SMS Sent',1)


# In[112]:


#BUILDING MODEL #3
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[113]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[114]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[115]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[116]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[117]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[118]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[119]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[120]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[121]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[122]:


# Let us calculate specificity
TN / float(TN+FP)


# In[123]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[124]:


# positive predictive value 
print (TP / float(TP+FP))


# In[125]:


# Negative predictive value
print (TN / float(TN+ FN))


# # PLOTTING ROC CURVE
# 

# In[126]:


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


# In[127]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[128]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# # The ROC Curve should be a value close to 1. We are getting a good value of 0.98 indicating a good predictive model.
# 
# 

# # Finding Optimal Cutoff Point¶
# 

# Above we had chosen an arbitrary cut-off value of 0.5. 
# We need to determine the best cut-off value and the below section deals with that:
# 
# 

# In[130]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[131]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
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


# In[132]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[133]:


#### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[134]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[135]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[136]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[137]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[138]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[139]:


# Let us calculate specificity
TN / float(TN+FP)


# Accuracy 93.67%
# Sensitivity 94.16%
# Specificity 93.37%

# In[140]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[141]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[142]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[143]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[144]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[145]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[146]:


from sklearn.metrics import precision_score, recall_score


# In[147]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[148]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[149]:


from sklearn.metrics import precision_recall_curve


# In[150]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[151]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[152]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[153]:


X_test = X_test[col]
X_test.head()


# In[154]:


X_test_sm = sm.add_constant(X_test)


# In[155]:


y_test_pred = res.predict(X_test_sm)


# In[156]:


y_test_pred[:10]


# In[160]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)

y_pred_1.head()


# In[161]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[162]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[163]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[165]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[166]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
y_pred_final.head()


# In[167]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[168]:


y_pred_final.head()


# In[169]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[170]:


y_pred_final.head()


# In[171]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[172]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[173]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[174]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[175]:


# Let us calculate specificity
TN / float(TN+FP)


# In[176]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[177]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# Accuracy 93.47%
# Sensitivity 94.79%
# Specificity 92.60%

# # Final Observation

# 
# Let us compare the values obtained for Train & Test:
# 
# Train Data: 
# Accuracy : 93.67%
# Sensitivity : 94.16%
# Specificity : 93.37%
# Test Data: 
# Accuracy : 93.47%
# Sensitivity : 94.79%
# Specificity : 92.60%
# 

# In[ ]:




