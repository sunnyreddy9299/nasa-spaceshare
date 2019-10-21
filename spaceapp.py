
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
location1='C:\\Users\\Yadav\\Desktop\\nasa space challenge\\Fireball_And_Bollide_Reports.csv'
l1='C:\\Users\\Yadav\\Desktop\\f1.csv'
location2='C:\\Users\\Yadav\\Desktop\\nasa space challenge\\GlobalLandslideCataalog03122015.csv'
l2='C:\\Users\\Yadav\\Desktop\\f2.csv'
location3='C:\\Users\\Yadav\\Desktop\\nasa space challenge\\MeteoriteLandings.csv'
l3='C:\\Users\\Yadav\\Desktop\\f3.csv'
location4='C:\\Users\\Yadav\\Desktop\\nasa space challenge\\Near-Earth_Comets_-_Orbital_Elements.csv'
l4='C:\\Users\Yadav\\Desktop\\f4.csv'
data1=pd.read_csv(l1)
data2=pd.read_csv(l2)
data3=pd.read_csv(l3)
data4=pd.read_csv(l4)
data1.isnull()
#print(data1.isnull().tail())
data2


# In[56]:


data2.head()
w=data2.mean()
w
data2.isnull().sum()
data22=data2.fillna(w)
#data22=data22.dropna()
data22.isnull().sum()
data22.to_csv('C:\\Users\\Yadav\\Desktop\\f22.csv')
data22.isnull().sum()


# In[57]:


data3.columns


# In[58]:


data3.head()
data3.isnull().sum()
#x=120
x=data3.mean()
data33=data3.fillna(x)
data33=data3.dropna()
data33.isnull().sum()
data33.count()
#data33=pd.get_dummies(data33)
data33
#data33.columns
data33.isnull().sum()
data33.count()
data33.to_csv("C:\\Users\\Yadav\\Desktop\\f33.csv")


# In[59]:


data4.head()
data4.isnull().sum()
y=data4.mean()
data44=data4.fillna(y)
#data44
data44.isnull().sum()
data44.to_csv("C:\\Users\\Yadav\\Desktop\\f44.csv")


# In[60]:


data1.isnull().sum() 


# In[61]:


data1.columns


# In[62]:


data1.head()
ed=data1.mean()
data11=data1.fillna(ed)
data11
data11.isnull().sum() 
data11.to_csv("C:\\Users\\Yadav\\Desktop\\f11.csv")


# In[63]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
location1='C:\\Users\\Yadav\\Desktop\\f11.csv'
location2='C:\\Users\\Yadav\\Desktop\\f22.csv'
location3='C:\\Users\\Yadav\\Desktop\\f33.csv'
location4='C:\\Users\Yadav\\Desktop\\f44.csv'
d1=pd.read_csv(location1)
d2=pd.read_csv(location2)
d3=pd.read_csv(location3)
d4=pd.read_csv(location4)


# In[64]:


data1.columns
data2


# In[65]:


Xdata1=data1[['Altitude (km)','Velocity Components (km/s): vx','Velocity Components (km/s): vy','Velocity Components (km/s): vz']]
print(Xdata1.columns)
Ydata1=data1[['Calculated Total Impact Energy (kt)']]
print(Ydata1.columns)
Xdata2=data2[['latitude','longitude','population','distance']]
Ydata2=data2[['fatalities']]
Xdata3=data3[['reclat','reclong','mass (g)']]
Ydata3=data3[['recclass']]
Xdata4=data4[['e','i (deg)','w (deg)','A1 (AU/d^2)','A2 (AU/d^2)','A3 (AU/d^2)']]
Ydata4=data4[['DT (d)']]
Xd1=d1[['Altitude (km)','Velocity Components (km/s): vx','Velocity Components (km/s): vy','Velocity Components (km/s): vz']]
Yd1=d1[['Calculated Total Impact Energy (kt)']]
Xd2=d2[['latitude','longitude','population','distance','injuries','fatalities']]
Yd2=d2[['landslide_']]
Xd3=d3[['reclat','reclong','mass (g)']]
Yd3=d3[['recclass']]
Xd4=d4[['e','i (deg)','w (deg)','A1 (AU/d^2)','A2 (AU/d^2)','A3 (AU/d^2)']]
Yd4=d4[['Node (deg)']]
Yd2


# In[70]:


import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
Xtraindata1,Xtestdata1,Ytraindata1,Ytestdata1=train_test_split(Xdata1,Ydata1,test_size=0.33)
Xtraindata2,Xtestdata2,Ytraindata2,Ytestdata2=train_test_split(Xdata2,Ydata2,test_size=0.33)
Xtraindata2,Xtestdata3,Ytraindata3,Ytestdata3=train_test_split(Xdata3,Ydata3,test_size=0.33)
Xtraindata4,Xtestdata4,Ytraindata4,Ytestdata4=train_test_split(Xdata4,Ydata4,test_size=0.33)
Xtraind1,Xtestd1,Ytraind1,Ytestd1=train_test_split(Xd1,Yd1,test_size=0.33)
Xtraind2,Xtestd2,Ytraind2,Ytestd2=train_test_split(Xd2,Yd2,test_size=0.33)
Xtraind3,Xtestd3,Ytraind3,Ytestd3=train_test_split(Xd3,Yd3,test_size=0.33)
Xtraind4,Xtestd4,Ytraind4,Ytestd4=train_test_split(Xd4,Yd4,test_size=0.33)
'''
model1=RandomForestRegressor()
model1.fit(Xtraindata1,Ytraindata1)
a1=model1.predict(Xtestdata1)
print(a1)
model2=RandomForestClasifier()
model2.fit(Xtraindata2,Ytraindata2)
a2=model2.predict(Xtestdata2)
print(a2)
model3=RandomForestClassifier()
model3.fit(Xtraindata3,Ytraindata3)
a3=model3.predict(Xtestdata3)
print(a3)
model4=RandomForestRegressor()
model4.fit(Xtraindata4,Ytraindata4)
a4=model4.predict(Xtestdata4)
print(a4)
'''
model5=RandomForestRegressor()
model5.fit(Xtraind1,Ytraind1)
a5=model5.predict(Xtestd1)
print(a5)
rmse1=mean_squared_error(Ytestd1,a5)
print("the rmse value is")
print(rmse1)
model6=RandomForestClassifier()
Ytraind2=np.array(Ytraind2)
model6.fit(Xtraind2,Ytraind2)
a6=model6.predict(Xtestd2)
print(a6)
cf1=confusion_matrix(Ytestd2,a6)
print("confusion matrix is")
print(cf1)
model7=RandomForestClassifier()
model7.fit(Xtraind3,Ytraind3)
a7=model7.predict(Xtestd3)
print(a7)
cf2=confusion_matrix(Ytestd3,a7)
print("the cf2 value is")
print(cf2)
model8=RandomForestRegressor()
model8.fit(Xtraind4,Ytraind4)
a8=model8.predict(Xtestd4)
print(a8)
rmse2=mean_squared_error(Ytestd4,a8)
print("rmse2 is")
print(cf2)


# In[68]:


d2

