#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[3]:


data = pd.read_csv(r"C:\Users\Acer\OneDrive\Documents\train.csv")


# ### 1.Display the sample records

# In[4]:


print("Sample Records:")
print(data.head())


# ### 2 Find out the number of records and features, display numeric description of features

# In[6]:


num_records, num_features = data.shape
print(f"Number of Records: {num_records}")
print(f"Number of Features: {num_features}")
print("Numeric Description of Features:")
print(data.describe())


# ### 3 Identify and display binary features

# In[7]:


binary_features = [col for col in data.columns if data[col].nunique() == 2]
print("Binary Features:", binary_features)


# ### 4. Visualize number of records in each category of binary features

# In[8]:


for feature in binary_features:
    plt.figure(figsize=(6, 4))
    data[feature].value_counts().plot(kind='bar', title=feature)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()


# ### 5 Check for the missing values and replace it

# In[9]:


data.fillna(data.mean(), inplace=True)


# ### 6 Screen width

# In[31]:


zero_width_indices = data[data['px_width'] == 0].index
data.drop(zero_width_indices, inplace=True)


# ### 7 Identify the target feature and display the records

# In[33]:


target_feature = 'talk_time'
class_counts = data[target_feature].value_counts()
print("Class-wise Record Counts:")
print(class_counts)


# ### 8 

# In[34]:


X = data.drop(columns=[target_feature])
y = data[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)


# ### 9

# In[35]:


y_pred = knn_classifier.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs. Predicted:")
print(comparison)


# ### 10

# In[36]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[39]:


y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




