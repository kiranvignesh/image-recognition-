#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.simplefilter("ignore")


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[5]:


kit=os.listdir("C:/Users/kiran vignesh/OneDrive/Desktop/kit")


# In[6]:


kobe=os.listdir("C:/Users/kiran vignesh/OneDrive/Desktop/kobe")
emma=os.listdir("C:/Users/kiran vignesh/OneDrive/Desktop/emma")


# In[7]:


limit=10
kit_images=[None]*limit
j=0
for i in kit:
    if(j<limit):
        kit_images[j]=imread("C:/Users/kiran vignesh/OneDrive/Desktop/kit/"+i)
        j+=1
    else:
        break


# In[8]:


imshow(kit_images[0])


# In[9]:


limit=10
kobe_images=[None]*limit
j=0
for i in kobe:
    if(j<limit):
        kobe_images[j]=imread("C:/Users/kiran vignesh/OneDrive/Desktop/kobe/"+i)
        j+=1
    else:
        break


# In[10]:


imshow(kobe_images[0])


# In[11]:


limit=10
emma_images=[None]*limit
j=0
for i in emma:
    if(j<limit):
        emma_images[j]=imread("C:/Users/kiran vignesh/OneDrive/Desktop/emma/"+i)
        j+=1
    else:
        break


# In[12]:


imshow(emma_images[0])


# In[13]:


kit_gray=[None]*limit
j=0
for i in kit:
    if(j<limit):
        kit_gray[j]=rgb2gray(kit_images[j])
        j+=1
    else:
        break


# In[14]:


kobe_gray=[None]*limit
j=0
for i in kobe:
    if(j<limit):
        kobe_gray[j]=rgb2gray(kobe_images[j])
        j+=1
    else:
        break


# In[15]:


emma_gray=[None]*limit
j=0
for i in emma:
    if(j<limit):
        emma_gray[j]=rgb2gray(emma_images[j])
        j+=1
    else:
        break


# In[16]:


imshow(kit_gray[0])


# In[17]:


imshow(kobe_gray[0])


# In[18]:


imshow(emma_gray[0])


# In[19]:


kit_gray[3].shape


# In[20]:


kobe_gray[3].shape


# In[21]:


emma_gray[3].shape


# In[22]:


for j in range(10):
  ki=kit_gray[j]
  kit_gray[j]=resize(ki,(512,512))


# In[23]:


for j in range(10):
  ko=kobe_gray[j]
  kobe_gray[j]=resize(ko,(512,512))


# In[24]:


for j in range(10):
  em=emma_gray[j]
  emma_gray[j]=resize(em,(512,512))


# In[25]:


imshow(kit_gray[4])


# In[26]:


imshow(kobe_gray[4])


# In[27]:


imshow(emma_gray[4])


# In[28]:


len_of_images_kobe=len(kobe_gray)
len_of_images_kobe


# In[29]:


len_of_images_kit=len(kit_gray)
len_of_images_kit


# In[30]:


len_of_images_emma=len(emma_gray)
len_of_images_emma


# In[31]:


image_size_kit=kit_gray[1].shape
image_size_kit


# In[32]:


image_size_kobe=kobe_gray[1].shape
image_size_kobe


# In[33]:


image_size_emma=emma_gray[1].shape
image_size_emma


# In[34]:


flatten_size_kit=image_size_kit[0]*image_size_kit[1]
flatten_size_kit


# In[35]:


flatten_size_kobe=image_size_kobe[0]*image_size_kobe[1]
flatten_size_kobe


# In[36]:


flatten_size_emma=image_size_emma[0]*image_size_emma[1]
flatten_size_emma


# In[37]:


for i in range(len_of_images_kit):
  kit_gray[i]=np.ndarray.flatten(kit_gray[i]).reshape(flatten_size_kit,1)


# In[38]:


for i in range(len_of_images_kobe):
  kobe_gray[i]=np.ndarray.flatten(kobe_gray[i]).reshape(flatten_size_kobe,1)


# In[39]:


kit_gray=np.dstack(kit_gray)
kit_gray.shape


# In[40]:


kobe_gray=np.dstack(kobe_gray)
kobe_gray.shape


# In[41]:


emma_gray=np.dstack(emma_gray)
emma_gray.shape


# In[42]:


kit_gray=np.rollaxis(kit_gray,axis=2,start=0)
kit_gray.shape


# In[43]:


kobe_gray=np.rollaxis(kobe_gray,axis=2,start=0)
kobe_gray.shape


# In[44]:


emma_gray=np.rollaxis(emma_gray,axis=2,start=0)
emma_gray.shape


# In[45]:


kit_gray=kit_gray.reshape(len_of_images_kit,flatten_size_kit)
kit_gray.shape


# In[46]:


kobe_gray=kobe_gray.reshape(len_of_images_kobe,flatten_size_kobe)
kobe_gray.shape


# In[47]:


emma_gray=emma_gray.reshape(len_of_images_emma,flatten_size_emma)
emma_gray.shape


# In[48]:


kit_data=pd.DataFrame(kit_gray)
kit_gray


# In[49]:


kobe_data=pd.DataFrame(kobe_gray)
kobe_gray


# In[50]:


emma_data=pd.DataFrame(emma_gray)
emma_gray


# In[51]:


kit_data["label"]="kit"
kit_data


# In[52]:


kobe_data["label"]="kobe"
kobe_data


# In[53]:


emma_data["label"]="emma"
emma_data


# In[54]:


actor_1=pd.concat([kit_data,emma_data])


# In[55]:


actor=pd.concat([actor_1,kobe_data])


# In[56]:


actor


# In[57]:


from sklearn.utils import shuffle
hollywood_indexed=shuffle(actor).reset_index()
hollywood_indexed


# In[58]:


hollywood_actors=hollywood_indexed.drop(["index"],axis=1)
hollywood_actors


# In[59]:


x=hollywood_actors.values[:,:-1]


# In[60]:


y=hollywood_actors.values[:,-1]


# In[61]:


x


# In[62]:


y


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[65]:


from sklearn import svm


# In[66]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[67]:


y_pred=clf.predict(x_test)


# In[68]:


y_pred


# In[69]:


for i in (np.random.randint(0,6,4)):
  predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
  plt.title("Predicted label: {0}".format(y_pred[i]))
  plt.imshow(predicted_images,interpolation="nearest",cmap="gray")
  plt.show()


# In[70]:


from sklearn import metrics


# In[71]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[72]:


accuracy


# In[73]:


from sklearn.metrics import confusion_matrix


# In[74]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:




