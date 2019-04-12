#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[23]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection  import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


nltk.download(['punkt', 'wordnet', 'stopwords'])


# In[25]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('labeled_messages', engine)


# In[26]:


# drop nan values
df.dropna(axis=0, how = 'any', inplace = True)

X = df['message']
y = df.iloc[:,4:].astype(int)


# ### 2. Write a tokenization function to process your text data

# In[27]:


def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[28]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[30]:


def generate_report(y_test, y_pred):
    
    metrics = []
    for i, column in enumerate(y.columns.values):
        accuracy = accuracy_score(y_test[:,i], y_pred[:,i])
        precision = precision_score(y_test[:,i], y_pred[:,i], average='micro')
        recall = recall_score(y_test[:,i], y_pred[:,i], average='micro')
        f1 = f1_score(y_test[:,i], y_pred[:,i], average='micro')
        
        metrics.append([accuracy, precision, recall, f1])
        
    df = pd.DataFrame(data = np.array(metrics), index=y.columns.values, columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])
    return df


# In[31]:


# Evaluate training set
y_train_pred = pipeline.predict(X_train)


# In[32]:


generate_report(np.array(y_train), y_train_pred)


# In[33]:


y_test_pred = pipeline.predict(X_test)


# In[34]:


generate_report(np.array(y_test), y_test_pred)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[35]:


RandomForestClassifier().get_params()


# In[36]:


parameters = {
    'vect__min_df': [1, 5],
    'tfidf__use_idf':[True, False],
    'clf__estimator__n_estimators':[10, 25], 
    'clf__estimator__min_samples_split':[2, 5, 10]
}

cv = GridSearchCV(pipeline, param_grid=parameters)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[37]:


cv_model = cv.fit(X_train, y_train)


# In[38]:


cv.best_params_


# In[39]:


y_test_pred_cv = cv.predict(X_test)


# In[40]:


generate_report(np.array(y_test), y_test_pred_cv)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[41]:


pipeline2 = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])

parameters2 = {'vect__min_df': [5],
              'tfidf__use_idf':[True],
              'clf__estimator__learning_rate': [0.5, 1], 
              'clf__estimator__n_estimators':[10, 25]}

cv2 = GridSearchCV(pipeline2, param_grid=parameters2)


# In[42]:


AdaBoostClassifier().get_params()


# In[43]:


cv2_model = cv2.fit(X_train, y_train)


# In[44]:


y_test_pred_cv2 = cv2.predict(X_test)


# In[45]:


generate_report(np.array(y_test), y_test_pred_cv2)


# ### 9. Export your model as a pickle file

# In[46]:


with open('classifer.pkl', 'wb') as f:
    pickle.dump(cv2, f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.
