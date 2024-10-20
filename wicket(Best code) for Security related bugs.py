#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
full_df=pd.read_csv('Wicket.csv')


# In[2]:


full_df.head(1)
len(full_df)


# # DROP NULL VALUES 
# full_df.dropna(inplace = True) 
len(full_df)
# In[3]:


to_remove = full_df[(full_df['description'] == 0.0) | (pd.isnull(full_df['description']))]
df = full_df[~full_df['key'].isin(to_remove['key']) ]
print('before/after =',len(full_df), '/', len(df))


# In[4]:


to_remove = df[(df['summary'] == 0.0) | (pd.isnull(df['summary']))]
df1 = df[~df['key'].isin(to_remove['key']) ]
print('before/after =',len(df), '/', len(df1))

to_remove = df1[(df1['assignee'] == 'NaN') | (pd.isnull(df1['assignee']))]
df2 = df1[~df1['key'].isin(to_remove['key']) ]
print('before/after =',len(df1), '/', len(df2))
# In[5]:


df3= df1[df1['status.name'].isin(['Resolved'])]
print(df3['resolution'].unique())
print(len(df3))


# In[6]:


to_remove = full_df[(full_df['key'] == 0.0) | (pd.isnull(full_df['key']))]
df = full_df[~full_df['key'].isin(to_remove['key']) ]
print('before/after =',len(full_df), '/', len(df))


# In[7]:


#df3.columns
df3.columns


# In[8]:


print(len(df3))
df3.head(1)
print(len(df3))


# In[9]:


dfinal=df[['key','summary','description']]


# In[10]:


dfinal.head(1)


# In[11]:


issues=dfinal
dfinal=issues[pd.notnull(issues['description'])]
print('before/after =', len(issues),'/',len(dfinal))
dfinal.head()


# In[12]:


dfinal['combined']=dfinal['description']+dfinal['summary']


# In[13]:


dfinal.head(5)


# In[14]:


dfinals=dfinal.drop(['summary','description'],axis=1)


# In[15]:


dfinals.head(2)


# # TEXT MINING

# In[16]:


import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# # PREPROCESSING

# In[17]:


import re
def preprocess(text):
    text=text.lower()
    #text-re.sub("&lt;/?.*&gt;","&lt;*&gt; ",text)
    text=re.sub("(\\d|\\W)+" ," ", text)
    return text


# In[18]:


dfinals['combined']=dfinals['combined'].apply(lambda x:preprocess(x))


# In[19]:


dfinals.head(2)


# # LOWER CASE

# In[20]:


dfinals["text_lower"] = dfinals["combined"].str.lower()
dfinals.head()
#print(len(dfinals))


# # REMOVE PUNCTUATION
# 

# In[21]:


import string
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

dfinals["text_wo_punct"] = dfinals["text_lower"].apply(lambda text: remove_punctuation(text))
dfinals.head()


# In[22]:


dfinals.drop(["combined","text_lower"],axis=1,inplace=True)


# In[23]:


dfinals.head(2)


# In[24]:


from nltk.corpus import stopwords
", ".join(stopwords.words('english'))


# # REMOVE STOP WORDS

# In[25]:


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

dfinals["text_wo_stop"] = dfinals["text_wo_punct"].apply(lambda text: remove_stopwords(text))
dfinals.head()


# In[ ]:





# In[26]:


from collections import Counter
cnt = Counter()
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

dfinals["text_wo_stopfreq"] = dfinals["text_wo_stop"].apply(lambda text: remove_freqwords(text))
dfinals.head()


# In[27]:


dfinals.drop(["text_wo_punct","text_wo_stop"],axis=1,inplace=True)


# In[28]:


dfinals.head(2)


# # STEMMING

# In[29]:


from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

dfinals["text_stemmed"] = dfinals["text_wo_stopfreq"].apply(lambda text: stem_words(text))
dfinals.head()


# In[30]:


from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words

#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(dfinals['text_stemmed'])


# In[31]:


text_counts


# In[32]:


list(cv.vocabulary_.keys())[:10]


# In[33]:


from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
word_list=['security', 'vulnerablity', 'improper', 'neutralization', 'crosssite', 'bounds', 'memorybuffer', 
 'sqlinjection', 'injection', 'exposure', 'sensitive', 'unauthorized', 'forgery', 'overflow', 
 'wraparound', 'limitation', 'nullpointer', 'dereference', 'unrestricted', 'dangerous', 'incorrect', 
 'external', 'untrusted', 'uncontrolled', 'hard', 'deserialization', 'crash', 'attack', 'xss', 
 'fuzzing', 'corruption', 'invalid', 'access', 'authentication', 'permission', 'credentials', 
 'authorization', 'signature', 'encryption', 'safe', 'protected', 'audit', 'cve', 'detected', 
 'validation', 'restriction']

newlist=[]
for word in word_list:
    newlist.append(stemmer.stem(word))
newlist
pattern = '|'.join(newlist) 
dfinals['NEWcolumn'] = dfinals['text_stemmed'].str.contains(pattern)
dfinals['NEWcolumn'] = dfinals['NEWcolumn'].map({True: 'SBR', False: 'NSBR'})


# In[34]:


dfinals


# In[35]:


print(dfinals.loc[dfinals['NEWcolumn'].isin(['SBR'])])


# In[36]:


dfinals.to_csv("WicketSBRBugs.csv", encoding='utf-8', header=True, index=False)


# In[37]:


print(len(dfinals))


# In[38]:


dfi=dfinals.loc[dfinals['NEWcolumn'] == 'SBR']


# In[39]:


dfi


# In[40]:


print(dfinals.loc[dfinals['NEWcolumn'].isin(['SBR'])])



# In[41]:


dfinals.to_csv("WicketSBR Bugs.csv", encoding='utf-8', header=True, index=False)


# In[1]:


import pandas as pd
df=pd.read_csv('WicketSBR Bugs.csv')
df.head(10)

df['NEWcolumn'].value_counts()
df['NEWcolumn']=df['NEWcolumn'].apply(lambda x: 0 if x=='SBR' else 1)
df.head(10)
df.dropna(inplace = True) 
#len(full_df)
df=df.drop(['text_wo_stopfreq'],axis=1)


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['text_stemmed'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
text_tf, df['NEWcolumn'], test_size=0.3, random_state=123)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['NEWcolumn'], test_size=0.3, random_state=1)


# In[3]:


# from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
rf_probs=clf.predict(X_test)

ns_probs = [0 for _ in range(len(y_test))]
rf_probs = clf.predict_proba(X_test)[:, 1]

#lr_probs = lr_probs(X_test)[:, 1]
ns_auc = roc_auc_score(y_test, ns_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
# summarize scores
print('No bugs: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No bugs')
pyplot.plot(rf_fpr, rf_tpr, marker='_', label='RF')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
print('AUC: {:.2f}%'.format(roc_auc_score(y_test, rf_probs) * 100))

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print ('R Squared =',r2_score(y_test, y_pred))
print ('MAE =',mean_absolute_error(y_test, y_pred))
print ('MSE =',mean_squared_error(y_test, y_pred))
print('LOGLOSS Value is',log_loss(y_test, y_pred))



# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, accuracy_score
)
from sklearn.datasets import make_classification

# Generate a synthetic dataset (replace with your actual data)
#X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split dataset into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['NEWcolumn'], test_size=0.3, random_state=1)
# Define classifiers and their hyperparameters
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(probability=True),
    'GradientBoosting': GradientBoostingClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    'NaiveBayes': GaussianNB()
}

params = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'NaiveBayes': {}
}

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print the results
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}" if roc_auc is not None else "AUC-ROC: N/A")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Loop through classifiers and hyperparameter tuning
best_models = {}
for clf_name in classifiers:
    print(f"\nTuning {clf_name}...\n")
    
    clf = classifiers[clf_name]
    param_grid = params[clf_name]
    
    # Perform hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[clf_name] = best_model
    
    print(f"Best parameters for {clf_name}: {grid_search.best_params_}\n")
    
    # Evaluate the best model
    print(f"Performance metrics for {clf_name}:")
    evaluate_model(best_model, X_test, y_test)

# Optional: choose the best model based on F1-score or any other criteria
best_f1 = 0
best_clf_name = None

for clf_name, model in best_models.items():
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_clf_name = clf_name

print(f"\nBest model overall based on F1-Score: {best_clf_name} with F1-Score: {best_f1:.4f}")


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, accuracy_score
)
import math

# Assuming text_tf and df['NEWcolumn'] are already defined
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['NEWcolumn'], test_size=0.3, random_state=1
)

# Define classifiers and their hyperparameters
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(probability=True),
    'NaiveBayes': GaussianNB(),  # No hyperparameters to tune
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    'Bagging': BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=50)
}

params = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'NaiveBayes': {},  # No hyperparameters
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Bagging': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0]
    }
}

# Function to calculate FPR from confusion matrix
def calculate_fpr(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn)
    return fpr * 100  # FPR in percentage

# Function to calculate Balance (Bal)
def calculate_balance(fpr, recall):
    euclidean_distance = math.sqrt((0 - fpr)**2 + (100 - recall)**2)
    balance = 100 - (euclidean_distance / math.sqrt(2))
    return balance

# Function to evaluate model and store metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate FPR and Balance
    fpr = calculate_fpr(conf_matrix)
    balance = calculate_balance(fpr, recall * 100)
    
    # Return metrics as a dictionary
    return {
        "F1-Score": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC-ROC": roc_auc if roc_auc is not None else "N/A",
        "MCC": mcc,
        "Accuracy": accuracy,
        "FPR": fpr,
        "Balance": balance,
        "Confusion Matrix": conf_matrix
    }

# Create a list to hold metrics for each classifier
results = []

# Loop through classifiers and hyperparameter tuning
best_models = {}
for clf_name in classifiers:
    print(f"\nTuning {clf_name}...\n")
    
    clf = classifiers[clf_name]
    param_grid = params[clf_name]
    
    if clf_name == 'NaiveBayes':
        # Convert sparse matrix to dense for Naive Bayes
        X_train_dense = X_train.toarray()  # Convert training data to dense
        X_test_dense = X_test.toarray()    # Convert test data to dense
        
        try:
            clf.fit(X_train_dense, y_train)
            best_model = clf
        except ValueError as e:
            print(f"Error fitting Naive Bayes: {e}")
            continue  # Skip this classifier
    else:
        # Perform hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(clf, param_grid, scoring='f1', cv=5, n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

    best_models[clf_name] = best_model
    
    if clf_name != 'NaiveBayes':
        print(f"Best parameters for {clf_name}: {grid_search.best_params_}\n")
    
    # Evaluate the best model and store results
    print(f"Performance metrics for {clf_name}:")
    
    # Use dense data for Naive Bayes, sparse for others
    if clf_name == 'NaiveBayes':
        metrics = evaluate_model(best_model, X_test_dense, y_test)  # Pass dense test data
    else:
        metrics = evaluate_model(best_model, X_test, y_test)
    
    # Add metrics to results list
    results.append({
        "Classifier": clf_name,
        "F1-Score": metrics["F1-Score"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "AUC-ROC": metrics["AUC-ROC"],
        "MCC": metrics["MCC"],
        "Accuracy": metrics["Accuracy"],
        "FPR": metrics["FPR"],
        "Balance": metrics["Balance"],
    })
    
    # Print the results
    print(metrics)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to an Excel file
results_df.to_excel("classifier_performance_metrics.xlsx", index=False)

print("Performance metrics saved to 'classifier_performance_metrics.xlsx'.")

# Optional: choose the best model based on F1-score or any other criteria
best_f1 = 0
best_clf_name = None

for clf_name, model in best_models.items():
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_clf_name = clf_name

print(f"\nBest model overall based on F1-Score: {best_clf_name} with F1-Score: {best_f1:.4f}")


# In[ ]:





# In[ ]:




