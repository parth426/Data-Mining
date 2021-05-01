#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning - ResNet50

# In[1]:


import tensorflow as tf


# In[2]:


print(tf.__version__)


# In[3]:


import keras


# In[4]:


print(keras.__version__)


# In[5]:


from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import load_model,Model


# ##### ImageDataGenerator helps us to perform Data Augmentation

# In[6]:


from IPython import get_ipython


# In[7]:


import numpy as np
from glob import glob
import matplotlib.pyplot as plt
# %matplotlib inline


# glob (short for global) is used to return all file paths that match a specific pattern

# In[8]:


img_size = [224,224]


# In[9]:


train_path = 'Datasets/Train'


# In[10]:


test_path = 'Datasets/Test'


# In[11]:


resnet = ResNet50(input_shape=img_size+[3],weights='imagenet',include_top=False)


# In[12]:


resnet.summary()


# In[13]:


# Do not train existing weights
for layer in resnet.layers:
    layer.trainable = False


# ###### glob is useful to get O/P classes

# In[14]:


folders = glob('Datasets\\Train\\*')


# In[15]:


folders


# In[16]:


len(folders)


# In[17]:


x = Flatten()(resnet.output)


# In[18]:


prediction = Dense(len(folders),activation='softmax')(x)


# In[19]:


model = Model(inputs = resnet.input, outputs=prediction)


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[22]:


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[23]:


train_data = train_datagen.flow_from_directory('Datasets/Train',
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode='categorical'
                                              )


# In[24]:


test_data = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')


# In[25]:


r = model.fit_generator(train_data,
                       validation_data=test_data,
                       epochs=50,
                       steps_per_epoch = len(train_data),
                       validation_steps=len(test_data))


# In[26]:


# Plot loss
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[27]:


plt.plot(r.history['accuracy'], label='Train Acc')
plt.plot(r.history['val_accuracy'],label='Validation Accuracy')
plt.legend()
plt.show()


# In[28]:


r.history


# ## Saving the model

# In[29]:


from keras.models import load_model


# In[30]:


model.save('resnet50_car.h5')


# In[31]:


y_pred = model.predict(test_data)


# In[32]:


y_pred


# In[33]:


y_pred = np.argmax(y_pred,axis=1)


# In[34]:


y_pred


# In[35]:


model = load_model('resnet50_car.h5')


# In[36]:


img = image.load_img('Datasets/Test/audi/22.jpg',target_size=(224,224))


# In[37]:


img


# In[38]:


x = image.img_to_array(img)


# In[39]:


x


# In[40]:


x = x/255
x


# In[41]:


x.shape


# In[42]:


x = np.expand_dims(x,axis=0)


# In[43]:


x.shape


# In[44]:


from keras.applications.resnet50 import preprocess_input


# In[45]:


img_data = preprocess_input(x)


# In[46]:


img_data


# In[47]:


img_data.shape


# In[48]:


model.predict(img_data)


# In[49]:


a = np.argmax(model.predict(img_data),axis=1)


# In[50]:


a


# In[ ]:





# In[ ]:





# In[ ]:




