
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_88e03573e47348bbb094133452e5642e = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='6ezsl4FXxJJoECLJx1QtrHR4h-AnGvfFgI4SFpupGguH',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_88e03573e47348bbb094133452e5642e.get_object(Bucket='smartinvestmentprediction-donotdelete-pr-j8gvygn1s4whna',Key='Smart Investment Prediction.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset= pd.read_csv(body)
dataset.head()



# # Import data set

# In[3]:


type(dataset)


# In[4]:


dataset.isnull() #finding null values


# In[5]:


dataset.isnull().any()


# # Separeting Independent and Dependent variables

# In[6]:


x=dataset.iloc[:,0:5]


# In[7]:


x


# In[8]:


y=dataset.iloc[:,5:]


# In[9]:


y


# In[10]:


x=dataset.iloc[:,0:5].values


# In[11]:


x


# In[12]:


y=dataset.iloc[:,5:].values


# In[13]:


y


# In[14]:


x.shape


# In[15]:


y.shape


# # splitting into Train and Test Datasets

# In[16]:


from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[17]:


x_train


# In[18]:


x_test


# In[19]:


y_train


# In[20]:


y_test


# # Using Standard Scaler

# In[21]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[22]:


x_train


# # Decision Tree 

# In[23]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0) #inplaces of entropy we can change name into "gini" so that we can calculate another method for decision


# In[24]:


classifier.fit(x_train,y_train)


# In[25]:


y_predict=classifier.predict(x_test)


# In[26]:


y_predict


# In[27]:


y_test


# In[28]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[29]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc1=metrics.auc(fpr,tpr)
roc_auc1


# In[30]:


plt.plot(fpr,tpr,label='AUC=0%.2f'%roc_auc1)
plt.legend()
plt.show()


# # Random Forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier
classifier1=RandomForestClassifier(n_estimators=30,criterion='gini',random_state=0)


# In[32]:


classifier1.fit(x_train,y_train)


# In[33]:


y_predict=classifier1.predict(x_test)


# In[34]:


y_predict


# In[35]:


y_test


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[37]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc2=metrics.auc(fpr,tpr)
roc_auc2


# In[38]:


plt.plot(fpr,tpr,label='AUC=0%.2f'%roc_auc2)
plt.legend()
plt.show()


# # K N Neighbor

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)


# In[40]:


classifier.fit(x_train,y_train)


# In[41]:


y_predict=classifier.predict(x_test)


# In[42]:


y_predict


# In[43]:


y_test


# In[44]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[45]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc3=metrics.auc(fpr,tpr)
roc_auc3


# In[46]:


plt.plot(fpr,tpr,label='AUC=0%.2f'%roc_auc3)
plt.legend()
plt.show()


# # Confusion Matrix

# In[47]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
cm


# In[48]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc4=metrics.auc(fpr,tpr)
roc_auc4


# In[49]:


plt.plot(fpr,tpr,label='AUC=%0.2f'%roc_auc4)
plt.title('Confusion Matrix')
plt.legend()
plt.show


# # Bar Graph

# In[50]:


x=['Decision Tree','Random Forest','KN Neighbor','Confusion Matrix']
y=[roc_auc1,roc_auc2,roc_auc3,roc_auc4]
plt.title('Investment Prediction Accuracy')
plt.bar(x,y,width=0.4)


# In[51]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[52]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[53]:


wml_credentials={"instance_id": "ff99982a-49b0-45d0-9c28-988c1f4ea51f",
  "password": "0fb600ab-da2e-4932-91d4-b3cc41625adf",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "8ce86d86-bc0d-47b4-b099-618aa5b846df","access_key": "BYhzJ3mQCnIeJKaBvBj6O8-LUkIcYNC20aUumPUw8gjO"}


# In[54]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[55]:


import json
instance_details= client.service_instance.get_details()
print(json.dumps(instance_details,indent=2))


# In[56]:


model_props={client.repository.ModelMetaNames.AUTHOR_NAME:"kasturi",
             client.repository.ModelMetaNames.AUTHOR_EMAIL:"mudidekasturi18@gmail.com",
             client.repository.ModelMetaNames.NAME:"SIPrediction"}


# In[57]:


model_artifact =client.repository.store_model(classifier1,meta_props=model_props)


# In[58]:


published_model_uid=client.repository.get_model_uid(model_artifact)


# In[59]:


published_model_uid


# In[60]:


created_deployments=client.deployments.create(published_model_uid,name="SIPrediction")


# In[61]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployments)


# In[62]:


scoring_endpoint


# In[63]:


client.deployments.list()

