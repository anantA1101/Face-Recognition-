from  keras.layers import Input, Lambda, Dense, Flatten 
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# resizing the input image
IMAGE_SIZE = [224, 224] 

train_path = r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\images\train/'
valid_path = r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\images\validation/'

# add preprossesing layer infront of vgg16 algo
vgg = VGG16(input_shape= IMAGE_SIZE+[3], weights='imagenet',include_top= False)

# dont train existing weights
for layer in vgg.layers:
    layer.trainable= False

# usefull for getting number classes 
folders = glob(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\images\train/*')


#making our own layers - u can add more layers
x= Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object 
model= Model(inputs=vgg.input, outputs=prediction)

#view the structure of the model 
model.summary()

#telling the model what cost and optimization method to use 
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

train_datagen= ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen= ImageDataGenerator(rescale=1./255)

#now creating training and test set
training_set = train_datagen.flow_from_directory(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\images\train/',
                                                 target_size=(224,224),
                                                 batch_size=8,
                                                 class_mode='categorical')

test_set= test_datagen.flow_from_directory(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\images\validation/',
                                           target_size=(224,224),
                                           batch_size=8,
                                           class_mode='categorical')

#fitting the modell
r= model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=4,
    steps_per_epoch=20, #len(training_set),
    validation_steps=20 #len(test_set)
    )
#loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\LossVal_loss')

#accuraciies 
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\AccVal_acc')

model.save(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\face_recognition.h5')