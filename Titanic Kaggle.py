#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train= pd.read_csv("C:\\Users\\Shubham\\Downloads\\Titanic\\train.csv")
train.head()


# In[3]:


train.shape


# In[4]:


train["Fare"].fillna(train.Fare.median(), inplace=True)
train["Embarked"].fillna(train.Embarked.mode()[0], inplace=True)


# In[5]:


train["Age"].fillna(train.Age.median(), inplace=True)


# In[6]:


drop_column=["Cabin"]
train.drop(drop_column, axis=1, inplace=True)


# In[7]:


train.describe()


# In[8]:


test= pd.read_csv("C:\\Users\\Shubham\\Downloads\\Titanic\\test.csv")


# In[9]:


test.head()


# In[10]:


test["Age"].fillna(test.Age.median(), inplace=True)


# In[11]:


test["Fare"].fillna(test.Fare.median(), inplace=True)


# In[12]:


drop_column=["Cabin"]
test.drop(drop_column, axis=1, inplace=True)


# In[13]:


print("Check data for Nan value in train data")
print(train.isnull().sum())


# In[14]:


print("check data for Nan Value in Test data")
print(test.isnull().sum())


# ### Feature Engineering
# 

# In[15]:


#Combine all the data together
all_data= [train, test]


# In[16]:


for dataset in all_data:
    dataset["Familysize"]= dataset["SibSp"]+dataset["Parch"]+1


# In[17]:


import re
#Extract title from passengers name
def get_title(name):
    title_search=re.search('([A-Za-z]+)\.', name)
    #if title exists return it
    if title_search:
        return title_search.group(1)
    return ""
for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    


# In[18]:


for dataset in all_data:
    dataset["Age_bin"]= pd.cut(dataset["Age"], bins=[0,12,21,40,120], labels=['Childrens','Teenager','Adult','Old'])


# In[19]:


for dataset in all_data:
    dataset["Fare_bin"]= pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_Fare','median_fare','average_fare','High_fare'])


# In[20]:


traindf=train
testdf=test


# In[21]:


all_dat=[traindf, testdf]


# In[22]:


traindf= pd.get_dummies(traindf, columns=["Sex","Title","Age_bin","Embarked","Fare_bin"] , prefix=["Sex","Title","Age_type","Em_type","Fare_type"])


# drop_column = ['PassengerId']
# traindf.drop(drop_column, axis=1, inplace = True)

# In[23]:


traindf.head()


# In[24]:


testdf = pd.get_dummies(testdf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])


# In[25]:


testdf.head()


# In[27]:


drop_column = ['Age','Name','Ticket','Fare']
testdf.drop(drop_column, axis=1, inplace = True)


# In[28]:


drop_column = ['Age','Name','Ticket','Fare']
traindf.drop(drop_column, axis=1, inplace = True)


# In[29]:


traindf.head()


# In[30]:


testdf.head()


# ### Co-relation between features

# In[40]:


import seaborn as sns


# In[45]:


sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# ### Model selection

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


# In[52]:


all_features = traindf.drop("Survived",axis=1)
Targeted_feature = traindf["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(all_features, Targeted_feature, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# ### Model
# 

# In[54]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[57]:


model=RandomForestClassifier(n_estimators=700,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)
model.fit(X_train, Y_train)


# In[59]:


prediction = model.predict(X_test)


# In[65]:


print ("Accuracy of RandomForest is:", accuracy_score(prediction, Y_test)*100)


# In[105]:


prediction.shape


# In[67]:


kFold=KFold(n_splits=10, random_state=22)
result = cross_val_score(model, all_features, Targeted_feature, cv=10, scoring="accuracy")
print("Cross Validated Score of RandomFOrest is:", result.mean()*100)


# In[103]:


y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)
y_pred


# In[106]:


sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='d', cmap="summer")
plt.title('Confusion matrix\n', y=1.05, size=15)


# In[122]:


drop_column=["PassengerId"]
traindf.drop(drop_column, axis=1, inplace=True)


# In[123]:


train_X = traindf.drop("Survived", axis=1)
train_Y = traindf["Survived"]
test_X = testdf.drop("PassengerId", axis=1).copy()


# In[124]:


train_X.shape, test_X.shape, train_Y.shape


# In[125]:


model = RandomForestClassifier()
n_estim=range(100,1000,100)

## Search grid for optimal parameters
param_grid = {"n_estimators" :n_estim}


model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_rf.fit(train_X,train_Y)



# Best score
print(model_rf.best_score_)

#best estimator
model_rf.best_estimator_


# In[132]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
random_forest.fit(train_X, train_Y)
Y_pred_rf = random_forest.predict(test_X)
random_forest.score(train_X,train_Y)
acc_random_forest = round(random_forest.score(train_X, train_Y) * 100, 2)

print("Important features")
pd.Series(random_forest.feature_importances_,train_X.columns).sort_values(ascending=True).plot.barh(width=0.8)
print('__'*30)
print(acc_random_forest)


# In[135]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_rf})

submission.to_csv("C:\\Users\\Shubham\\Downloads\\Titanic\\Titanic.csv", index=True)


# In[ ]:




