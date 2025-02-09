# # Megaline
# 
# ### The purpose of this project is to create a model that would evaluate the behavior of subscribers and suggest either Smart or Ultra, two of Megaline's most recent plans. This would boost the company's strength in the market and assist in driving more revenue.


# Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ### Libraries are imported.

# In[2]:


# Read the CSV file into a DataFrame
df = pd.read_csv("/datasets/users_behavior.csv")


# ### The above dataframe is for the Users Behavior of Megaline cellular company.

# In[3]:


# Working code

df.describe()


# ### There are 5 columns in this dataframe: Calls, Minutes, Messages, Mb_used, and is_ultra.

# In[4]:


# Working code

df.head()


# In[5]:


# Working code

df.isna().sum()


# In[6]:


# Splitting the Dataset

# working code

X = df[['calls', 'minutes', 'messages', 'mb_used']]
y = df['is_ultra']


# ### The Dataset is splitted into 2 parts, X and Y. Each contain different columns.

# 

# In[7]:


# Training and Testing sets (70% train, 30% test)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

#  training and validation sets (50% validation, 50% test)

X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

X_train.shape, X_valid.shape, X_test.shape


# ### The shapes of resulting datasets are: 2249 samples and 4 features each, 482 samples and 4 features each, 483 samples and 4 features each.

# In[8]:


# Scaling Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
X_train[:5]


# ### The first five samples in the training set's impacted values are displayed in this array. The training set's related feature's mean and standard deviation were used to modify the values.

# In[9]:


# Baseline model

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

# accuracy
np.mean(y_pred == y_valid)


# ### The model correctly predicted approximately 76.56% of the validation set instances.

# In[10]:


# Hyperparameters and their ranges for Decision Trees
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search_dt = GridSearchCV(estimator=dt_classifier, param_grid=param_grid_dt, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search_dt.fit(X_train, y_train)

# Get the best hyperparameters
best_params_dt = grid_search_dt.best_params_
best_dt_model = grid_search_dt.best_estimator_

# Predict on the test set
y_pred_dt = best_dt_model.predict(X_test)

# Print classification report
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Print confusion matrix
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:")
print(conf_matrix_dt)

# Evaluate the best model on the test set
accuracy_dt = best_dt_model.score(X_test, y_test)
print(f"Accuracy of the best Decision Tree model: {accuracy_dt:.2f}")


# ### Confusion Matrix: 
# 298: True negatives (class 0 correctly predicted).
# 21: False positives (class 0 incorrectly predicted as class 1).
# 89: False negatives (class 1 incorrectly predicted as class 0).
# 75: True positives (class 1 correctly predicted).
# 
# ### Decision Trees Accuracy: 
# The overall accuracy of the best Decision Tree model on the test set is 77%.

# In[11]:


#  Hyperparameters tuning for Linear Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300]
}
# Initialize GridSearchCV
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')

# Fit the model
grid_search_lr.fit(X_train, y_train)

# Get the best model
best_lr = grid_search_lr.best_estimator_

# Predict on the test set
y_pred_lr = best_lr.predict(X_test)

# Print classification report
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Print accuracy
accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy of the Linear Regression Model: {accuracy:.4f}")


# ### Linear Regression Accuracy: 
# The overall accuracy of the best Linear Regression model on the test set is 72.46%.

# In[ ]:


# Hyperparameters for RandomForest

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Get the best model
best_rf = grid_search_rf.best_estimator_
    
# Predict on the test set
y_pred_rf = best_rf.predict(X_test)

# Print classification report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Print accuracy
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy:.4f}")


# ### Random Forest Accuracy:
# The overall accuracy of the best Random Forest model on the test set is 78.67%.

# In[ ]:


# Decision Tree: Sanity Check

# Checking for Overfitting

# Evaluate the best model on the training set
accuracy_train_dt = accuracy_score(y_train, best_dt_model.predict(X_train))
print(f"Accuracy of the best Decision Tree model on the training set: {accuracy_train_dt:.2f}")

# Evaluate the best model on the test set
accuracy_test_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of the best Decision Tree model on the test set: {accuracy_test_dt:.2f}")


# In[ ]:


# Linear Regression: Sanity Check

# Checking for Overfitting

accuracy_train_lr = accuracy_score(y_train, best_lr.predict(X_train))
accuracy_test_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy of the Logistic Regression Model on the training set: {accuracy_train_lr:.4f}")
print(f"Accuracy of the Logistic Regression Model on the test set: {accuracy_test_lr:.4f}")


# In[ ]:


# Random Forest: Sanity Check

# Checking for Overfitting

accuracy_train_rf = accuracy_score(y_train, best_rf.predict(X_train))
accuracy_test_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of the Random Forest Model on the training set: {accuracy_train_rf:.4f}")
print(f"Accuracy of the Random Forest Model on the test set: {accuracy_test_rf:.4f}")


# # Conclusion
# 
# Random Forest had the highest accuracy at 78.67%, followed by Decision Tree at 77%, and Linear Regression at 72.46%.
# 
# In comparison to the other models, Random Forest frequently gave higher precision and recall for both classes, particularly for class 1.
# 
# When averaged across both classes, Random Forest obtained the highest F1-score, suggesting a better balance between recall and precision.
# 
# As a result of the Random Forest Machine Learning model have an accuracy score close to 79%, this would implicate that the stakeholders can put their faith in this model and trust it's ability of obtaining better results for the organization.
# 
# A risk of 21.34% remains valid until a solution have to be found such as improving highlight areas in the Model.
# 
# Implementing this Machine Learning model (Random Forest) helps in reducing cost and increasing efficiency.
# 
# Random Forest model will help Megaline management recommend their newer plans: Smart or Ultra.
