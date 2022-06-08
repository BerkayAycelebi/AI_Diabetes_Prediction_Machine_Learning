#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data=pd.read_csv("diabetes.csv")
data.head()


# In[17]:


plt.figure(figsize=(15,6))
diabetes = data[data.Outcome == 1]
healty =data[data.Outcome == 0]
plt.scatter(healty.Age, healty.Glucose,color="blue", label="Healty", alpha=0.4)
plt.scatter(diabetes.Age, diabetes.Glucose,color="red", label="Diabetes", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()
#Here we have drawn an example graph according to glucose
#Our Machine Learning Model will predict by looking at all the data


# In[26]:


#define to x and y axis
#y axis is Outcome values means healty or diabetes like boolean 
#x axis is all colums our dataset without Outcome,
#K-Nearest Neighbors will be classification in x_Crude_Datas given below code
y=data.Outcome.values
x_Crude_Data=data.drop(["Outcome"],axis=1)

#normalization
#Before training we should normalize to datas cause KNN algorithm can work wrong
x=(x_Crude_Data - np.min(x_Crude_Data))/(np.max(x_Crude_Data)-np.min(x_Crude_Data))
#Here we re trainin our model with datas %80 otherwise is for test part
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
#n_neighbors means is the number of how many neighbors we want the algorithm to look at
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
#here our test result
print("result",knn.score(x_test,y_test))



# In[32]:


#we find the optimal number of n_neighbors
cnt=1
for k in range(1,8):
    knn_new=KNeighborsClassifier(n_neighbors=k)
    knn_new.fit(x_train,y_train)
    print(cnt," ","Correctness rate %", knn_new.score(x_test,y_test)*100)
    cnt+=1


# In[38]:


#Prediction part
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc.fit_transform(x_Crude_Data)
#We make predictions with data that is not in the data set.
new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]


# In[ ]:




