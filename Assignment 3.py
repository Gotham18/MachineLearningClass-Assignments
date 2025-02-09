#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[127]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[128]:


player_df = pd.read_csv(r"C:\Users\GothamTikyani\Downloads\fifa19.csv")


# In[129]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[130]:


player_df = player_df[numcols+catcols]


# In[131]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[132]:


traindf = pd.DataFrame(traindf,columns=features)


# In[133]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[134]:


X.head()


# In[135]:


len(X.columns)


# ### Set some fixed set of features

# In[136]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[137]:


def cor_selector(X, y,num_feats):
    cor_list = []
    for i in X.columns:
        cor = np.corrcoef(X[i],y)[0,1]
        cor_list.append(abs(cor))
        
    cor_array = np.array(cor_list)
    
    cor_order = np.argsort(cor_array)[-num_feats:]
    
    cor_support = np.zeros(X.shape[1], dtype=bool)
    cor_support[cor_order]= True
    
    cor_feature = X.columns[cor_support]
    
   
    return cor_support, cor_feature


# In[138]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[139]:


print(list(cor_feature))


# ## Filter Feature Selection - Chi-Sqaure

# In[140]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[141]:


def chi_squared_selector(X, y, num_feats):
    chi_selector = SelectKBest(score_func=chi2, k=num_feats)
    chi_selector.fit(X, y)
    
    chi_support = chi_selector.get_support()
    
    chi_feature = X.columns[chi_support]
    
    # Your code ends here
    return chi_support, chi_feature


# In[142]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[143]:


print(list(chi_feature))


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[144]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[145]:


def rfe_selector(X, y, num_feats):

   
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
   
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    
    rfe = RFE(estimator, n_features_to_select=num_feats)
    

    rfe.fit(X_scaled, y)
    
    
    rfe_support = rfe.support_
    
    
    rfe_feature = X.columns[rfe_support]
    
    return rfe_support, rfe_feature


# In[146]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[147]:


list(rfe_feature)


# ## Embedded Selection - Lasso: SelectFromModel

# In[148]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[149]:


def embedded_log_reg_selector(X, y, num_feats):
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    estimator = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    
    
    selector = SelectFromModel(estimator, max_features=num_feats, threshold=-np.inf)  # Use all available coefficients
    selector.fit(X_scaled, y)
    
    
    embedded_lr_support = selector.get_support()
    
    
    embedded_lr_feature = X.columns[embedded_lr_support]
    return embedded_lr_support, embedded_lr_feature


# In[150]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[151]:


list(embedded_lr_feature)


# ## Tree based(Random Forest): SelectFromModel

# In[152]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[153]:


def embedded_rf_selector(X, y, num_feats):
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    selector = SelectFromModel(estimator, max_features=num_feats, threshold=-np.inf)
    selector.fit(X, y)
   
    embedded_rf_support = selector.get_support()
    
    embedded_rf_feature = X.columns[embedded_rf_support]
    
    return embedded_rf_support, embedded_rf_feature


# In[154]:


embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')


# In[155]:


list(embedded_rf_feature)


# ## Tree based(Light GBM): SelectFromModel

# In[156]:


get_ipython().system('pip install lightgbm')
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[157]:


def embedded_lgbm_selector(X, y, num_feats):
    estimator = LGBMClassifier(random_state=42)
    
    selector = SelectFromModel(estimator, max_features=num_feats, threshold=-np.inf)
    selector.fit(X, y)
   
    embedded_lgbm_support = selector.get_support()
    embedded_lgbm_feature = X.columns[embedded_lgbm_support]
    
    return embedded_lgbm_support, embedded_lgbm_feature


# embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
# print(str(len(embedded_lgbm_feature)), 'selected features')

# In[158]:


list(embedded_lgbm_feature)


# ## Putting all of it together: AutoFeatureSelector Tool

# In[159]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[160]:


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    df = pd.read_csv(r"C:\Users\GothamTikyani\Downloads\fifa19.csv")
    
    
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]   # Lets assume that the last column is the target variable
    
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    
    num_feats = 30  # Example value for now 
    
    # Your code ends here
    return X, y, num_feats


# In[164]:


from sklearn.preprocessing import StandardScaler
from collections import Counter

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
        all_selected_features.extend(cor_feature)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
        all_selected_features.extend(chi_feature)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
        all_selected_features.extend(embedded_rfe_feature)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        all_selected_features.extend(embedded_lr_feature)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        all_selected_features.extend(embedded_rf_feature)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        all_selected_features.extend(embedded_lgbm_feature)
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    feature_votes = Counter(all_selected_features)
    best_features = [feature for feature, count in feature_votes.most_common(num_feats)]
    #### Your Code ends here
    return best_features


# In[165]:


best_features = autoFeatureSelector(dataset_path= r"C:\Users\GothamTikyani\Downloads\fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# In[ ]:





# In[ ]:




