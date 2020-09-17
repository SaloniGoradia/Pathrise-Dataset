#!/usr/bin/env python
# coding: utf-8
#test
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas_profiling
import sys
import streamlit as st
#get_ipython().system('{sys.executable} -m pip install pandas-profiling')
import warnings
warnings.filterwarnings('ignore')

st.title("Pathrise Data Project")
data = pd.read_csv(r"C:/Users/sgoradia/Desktop/Python/Pathrise project/data_Pathrise.csv")
st.header("How does data look like?")
st.write(data.head())

st.header("Pathrise Data with only Active and Placed candidates")
data= data[(data["pathrise_status"] == "Active") | (data["pathrise_status"] == "Placed")]
st.write(data.head())

#Remove Serial Number and Cohort tag
data.drop("id", axis=1, inplace=True)
data.drop("cohort_tag", axis=1, inplace=True)
data.drop("pathrise_status", axis=1, inplace=True)

#description of data
st.header("Description of data")
st.write(data.describe())
data.info()

#st.header("Data visualisation: Profie report")
#a=data.profile_report(title='data analysis at pathrise', progress_bar=False)
#st.write(a.html,unsafe_allow_html = True)

#pairplot
st.header("Pairplots")
grid = sns.PairGrid(data= data, hue='placed')
grid = grid.map_upper(plt.scatter)
grid = grid.map_diag(sns.kdeplot, shade=True)
grid = grid.map_lower(sns.kdeplot)
#plt.title('Distribution of the features',loc='left')
st.pyplot(plt)

#stripping the column names
data.columns = data.columns.str.rstrip()

#plots for categorical variables
st.header("Barplots for categorical variables")
cat_feats = ['primary_track', 'employment_status', 'highest_level_of_education', 'length_of_job_search',
             'biggest_challenge_in_search', 'professional_experience','work_authorization_status','gender','race']

fig, axes = plt.subplots(3, 3, figsize=(20, 30))

sns.countplot(data.primary_track, hue=data.placed, ax=axes[0][0])
a=sns.countplot(data.employment_status, hue=data.placed, ax=axes[0][1])
a.set_xticklabels(a.get_xticklabels(), rotation=90)

b=sns.countplot(data.highest_level_of_education, hue=data.placed, ax=axes[0][2])
b.set_xticklabels(b.get_xticklabels(), rotation=90)

c=sns.countplot(data.length_of_job_search, hue=data.placed, ax=axes[1][0])
c.set_xticklabels(c.get_xticklabels(), rotation=90)

d=sns.countplot(data.biggest_challenge_in_search, hue=data.placed, ax=axes[1][1])
d.set_xticklabels(d.get_xticklabels(), rotation=90)

e=sns.countplot(data.professional_experience, hue=data.placed, ax=axes[1][2])
e.set_xticklabels(e.get_xticklabels(), rotation=90)

f=sns.countplot(data.work_authorization_status, hue=data.placed, ax=axes[2][0])
f.set_xticklabels(f.get_xticklabels(), rotation=90)

g=sns.countplot(data.gender, hue=data.placed, ax=axes[2][1])
g.set_xticklabels(g.get_xticklabels(), rotation=90)

h=sns.countplot(data.race, hue=data.placed, ax=axes[2][2])
h.set_xticklabels(h.get_xticklabels(), rotation=90)
st.pyplot(plt)

#boxplot
#st.header("Boxplot for numerical variables")
#sns.boxplot( x=data.placed, y=data.number_of_applications, width=0.5);
#plt.show()
#st.pyplot(plt)

#sns.boxplot( x=data.placed, y=data.number_of_interviews, width=0.5);
#plt.show()
#st.pyplot(plt)

##Data Preprocessing
##Feature encoding

st.header("Data Preprocessing: datatypes of data")
st.write(data.dtypes)

# count the number of missing values for each column
st.header("Number of missing values")
st.write(data[data.isnull().any(axis=1)])
##Missing values
data["gender"].isnull().sum()

##Imputing Missing values
data["employment_status"].fillna("Unemployed", inplace=True)
data["highest_level_of_education"].fillna("Some High School", inplace=True)
data["length_of_job_search"].fillna("Less than one month", inplace=True)
data["biggest_challenge_in_search"].fillna("Resume gap", inplace=True)
data["professional_experience"].fillna("Less than one year", inplace=True)
data["work_authorization_status"].fillna("Other", inplace=True)
data["number_of_interviews"].fillna(0, inplace=True)
data["number_of_applications"].fillna(0, inplace=True)
data["gender"].fillna("Decline to self identity", inplace=True)
data["race"].fillna("Decline to self identity", inplace=True)

data["employment_status"].isnull().sum()
data.dtypes

st.header("Creating different dataset with object variables")
obj_df = data.select_dtypes(include=['object']).copy()
st.write(obj_df.head())

#converting object into category
obj_df["primary_track"] = obj_df["primary_track"].astype('category')
obj_df["employment_status"] = obj_df["employment_status"].astype('category')
obj_df["highest_level_of_education"] = obj_df["highest_level_of_education"].astype('category')
obj_df["length_of_job_search"] = obj_df["length_of_job_search"].astype('category')
obj_df["biggest_challenge_in_search"] = obj_df["biggest_challenge_in_search"].astype('category')
obj_df["professional_experience"] = obj_df["professional_experience"].astype('category')
obj_df["work_authorization_status"] = obj_df["work_authorization_status"].astype('category')
obj_df["race"] = obj_df["race"].astype('category')
obj_df["gender"] = obj_df["gender"].astype('category')

data["primary_track"]=obj_df["primary_track"]
data["employment_status"]= obj_df["employment_status"]
data["highest_level_of_education"]=obj_df["highest_level_of_education"]
data["length_of_job_search"] =obj_df["length_of_job_search"]
data["biggest_challenge_in_search"] = obj_df["biggest_challenge_in_search"]
data["professional_experience"] = obj_df["professional_experience"]
data["work_authorization_status"] =obj_df["work_authorization_status"]
data["race"] = obj_df["race"]
data["gender"] =obj_df["gender"]

#data1 is copy of data, converting categories to int
data1= data.copy()
data1["primary_track"] = obj_df["primary_track"].cat.codes
data1["employment_status"] = obj_df["employment_status"].cat.codes
data1["highest_level_of_education"] = obj_df["highest_level_of_education"].cat.codes
data1["length_of_job_search"] = obj_df["length_of_job_search"].cat.codes
data1["biggest_challenge_in_search"] = obj_df["biggest_challenge_in_search"].cat.codes
data1["professional_experience"] = obj_df["professional_experience"].cat.codes
data1["work_authorization_status"] = obj_df["work_authorization_status"].cat.codes
data1["race"] = obj_df["race"].cat.codes
data1["gender"] = obj_df["gender"].cat.codes
st.header("Converting categorical variables to integer")
st.write(data1.head())

#Checking datatypes of latest data1
st.header("Datatypes of latest data")
st.write(data1.dtypes)

#for Classification
data_clf=data1.copy()
#for Regression
data_reg=data1.copy()

# Seperating Features and Target
st.header("Classification")
X = data_clf[['primary_track', 'employment_status', 'highest_level_of_education', 'length_of_job_search',
              'biggest_challenge_in_search', 'professional_experience', 'work_authorization_status','number_of_interviews',
              'number_of_applications', 'gender','race']]
st.write(X)
y = data_clf['placed']
st.write(y)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
st.header("Accuracy for DecisionTree")
st.write(accuracy_score(y_test, y_pred))

st.write(classification_report(y_test, y_pred))

#Using Random Forest Algorithm
st.header("RandomForestClassifier")
random_forest = RandomForestClassifier(n_estimators=100)
st.write(random_forest.fit(X_train, y_train))
y_pred = random_forest.predict(X_test)
st.write("Prediction of target variable on X_test")
st.write(y_pred)
st.write("Accuracy for RandomForest model")
st.write(accuracy_score(y_test, y_pred))
st.write(print(classification_report(y_test, y_pred)))

#Feature Importance
st.header("Feature Importance")
rows = list(X.columns)
imp = pd.DataFrame(np.zeros(6*len(rows)).reshape(2*len(rows), 3))
imp.columns = ["Classifier", "Feature", "Importance"]
#Add Rows
for index in range(0, 2*len(rows), 2):
    imp.iloc[index] = ["DecisionTree", rows[index//2], (100*dtree.feature_importances_[index//2])]
    imp.iloc[index + 1] = ["RandomForest", rows[index//2], (100*random_forest.feature_importances_[index//2])]

plt.figure(figsize=(15,5))
FI=sns.barplot("Feature", "Importance", hue="Classifier", data=imp)
FI.set_xticklabels(FI.get_xticklabels(), rotation=90)
plt.title("Computed Feature Importance")
st.pyplot(plt)

# Seperating Features and Target
X = data_clf[['employment_status', 'length_of_job_search',
              'biggest_challenge_in_search', 'professional_experience', 'work_authorization_status','number_of_interviews',
              'number_of_applications','gender','race']]
y = data_clf['placed']
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
accuracy_score(y_test, y_pred)

##Binary Classification with Logistic Regression
# Seperating Features and Target
st.header("Binary Classification with Logistic Regression")
X = data[['employment_status', 'highest_level_of_education', 'length_of_job_search',
              'biggest_challenge_in_search', 'professional_experience', 'work_authorization_status','number_of_interviews',
              'number_of_applications', 'gender','race']]
y = data['placed']

#One-Hot Encoding
X = pd.get_dummies(X)
colmunn_names = X.columns.to_list()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)
st.write("Prediction of target variable on X_test data")
st.write(y_pred)
st.write("Accuracy of LogisticRegression")
st.write(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(logistic_reg).fit(X_test, y_test)
eli5.show_weights(perm)

#get_ipython().system('pip install shap')
#get_ipython().system('pip install lime')
#get_ipython().system('pip install eli5')

st.header("Feature Importance using LogisticRegression")
plt.figure(figsize=(30, 10))
plt.bar(colmunn_names , perm.feature_importances_std_ * 100)
#plt.show()
st.pyplot(plt)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
##Regression Analysis
##Data preprocessing
#dropping NaNs (in Salary)
data_reg.dropna(inplace=True)
data_reg.drop("placed", axis=1, inplace=True)
#data_reg.drop("pathrise_status", axis=1, inplace=True)
st.header("Regression analysis for duration")
st.write(data_reg.head())
#Seperating Depencent and Independent Vaiiables
y = data_reg["program_duration_days"] #Dependent Variable
X = data_reg.drop("program_duration_days", axis=1)
column_names = X.columns.values
#One-Hot Encoding
X = pd.get_dummies(X)
colmunn_names = X.columns.to_list()
#Scalizing between 0-1 (Normalization)
X_scaled = MinMaxScaler().fit_transform(X)
#PDF of program duration days
sns.kdeplot(y)
#plt.show()
st.pyplot(plt)
#Selecting outliers
y[y > 400]
# 11 records
#Removing these Records from data
X_scaled = X_scaled[y < 400]
y = y[y < 400]
#pip install mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
linreg = LinearRegression()
sfs = SFS(linreg, k_features=1, forward=False, scoring='r2',cv=10)
sfs = sfs.fit(X_scaled, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Backward Elimination')
plt.grid()
#plt.show()
st.pyplot(plt)
#From Plot its clear that, many features actually decrease the performance
# Lets see the top 5 most significant features
top_n = 5
sfs.get_metric_dict()[top_n]
#Top N Features
top_n_indices = list(sfs.get_metric_dict()[top_n]['feature_idx'])
print(f"Most Significant {top_n} Features:")
for col in column_names[top_n_indices]:
    print(col)
#Select these Features only
X_selected = X_scaled[: ,top_n_indices]
lin_reg = LinearRegression()
lin_reg.fit(X_selected, y)
y_pred = lin_reg.predict(X_selected)
print(f"R2 Score: {r2_score(y, y_pred)}")
print(f"MAE: {mean_absolute_error(y, y_pred)}")

##Converting to DF for as  column names gives readibility
X_scaled = pd.DataFrame(X_scaled, columns=column_names)
y = y.values

# We must add a constants 1s for intercept before doing Linear Regression with statsmodel
X_scaled = sm.add_constant(X_scaled)
X_scaled.head()
#Constants 1 added for intercept term

# Step 1: With all Features
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()
# Identify max P-value (P>|t|) column
# Feature ssc_p has 0.995
#drop ssc_p
X_scaled = X_scaled.drop('length_of_job_search', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[221]:


# Identify max P-value (P>|t|) column
# Feature ssc_p has 0.995
#drop ssc_p
X_scaled = X_scaled.drop('highest_level_of_education', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[222]:


# Identify max P-value (P>|t|) column
# Feature ssc_p has 0.995
#drop ssc_p
X_scaled = X_scaled.drop('professional_experience', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:





# In[9]:


#Feature-Gender
data.gender.value_counts()


# In[10]:


sns.countplot("gender", hue="placed", data=data)
plt.show()


# In[12]:


sns.kdeplot(data.program_duration_days[ data.gender=="Male"])
sns.kdeplot(data.program_duration_days[ data.gender=="Female"])
plt.legend(["Male", "Female"])
plt.xlabel("program_duration_days")
plt.show()


# In[13]:


plt.figure(figsize =(18,6))
sns.boxplot("program_duration_days", "gender", data=data)
plt.show()


# In[14]:


#Feature-exp
data.professional_experience.value_counts()


# In[15]:


sns.countplot("professional_experience", hue="placed", data=data)
plt.show()


# In[16]:


sns.kdeplot(data.program_duration_days[ data.professional_experience=="1-2 years"])
sns.kdeplot(data.program_duration_days[ data.professional_experience=="Less than one year"])
sns.kdeplot(data.program_duration_days[ data.professional_experience=="3-4 years"])
sns.kdeplot(data.program_duration_days[ data.professional_experience=="5+ years"])
plt.legend(["1-2 years","Less than one year","3-4 years","5+ years"])
plt.xlabel("program_duration_days")
plt.show()


# In[18]:


#Feature-primary_track
data.primary_track.value_counts()


# In[19]:


sns.countplot("primary_track", hue="placed", data=data)
plt.show()


# In[21]:


sns.kdeplot(data.program_duration_days[ data.primary_track=="SWE"])
sns.kdeplot(data.program_duration_days[ data.primary_track=="PSO"])
sns.kdeplot(data.program_duration_days[ data.primary_track=="Design"])
sns.kdeplot(data.program_duration_days[ data.primary_track=="Data"])
plt.legend(["SWE","PSO","Design","Data"])
plt.xlabel("program_duration_days")
plt.show()


# In[22]:


plt.figure(figsize =(18,6))
sns.boxplot("program_duration_days", "primary_track", data=data)
plt.show()


# In[23]:


#Feature-highest_level_of_education
data.highest_level_of_education.value_counts()


# In[24]:


sns.countplot("highest_level_of_education", hue="placed", data=data)
plt.show()


# In[25]:


sns.kdeplot(data.program_duration_days[ data.highest_level_of_education=="Bachelor's Degree"])
sns.kdeplot(data.program_duration_days[ data.highest_level_of_education=="Master's Degree"])
sns.kdeplot(data.program_duration_days[ data.highest_level_of_education=="Some College, No Degree "])
sns.kdeplot(data.program_duration_days[ data.highest_level_of_education=="Doctorate or Professional Degree"])
plt.legend(["Bachelor's Degree","Master's Degree","Some College, No Degree ","Doctorate or Professional Degree"])
plt.xlabel("program_duration_days")
plt.show()


# In[26]:


sns.countplot("work_authorization_status", hue="placed", data=data)
plt.show()


# In[27]:
#abcd

#Kernel-Density Plot
sns.kdeplot(data.number_of_applications[ data.placed=="1"])
sns.kdeplot(data.number_of_applications[ data.placed=="0"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("number_of_applications")
plt.show()
