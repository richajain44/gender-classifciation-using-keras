
#importing all the necessary library
import numpy as np
import sklearn
import glob
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import shutil
from keras import applications
import keras.models
from keras.models import Model
from keras import optimizers

"""
Program to separate images into train and validation set
and then to create separate folders in train for female and male having equal number of images in both 
and also cresting same folder structure in the validation set
"""
#creating lists
img_train =[]
img_test=[]
idlist_train =[]
idlist_test=[]
img_split=[]
idlist_split=[]
img_train_split=[]
img_test_split=[]
train_fname=[]

#splitting the images into train and validation set
for im4 in glob.glob("/home/itadmin/c:temp/training/image/*.jpg"):
    img_split.append(im4)
img_train_split = img_split[0:7600]
print(len(img_train_split))
img_test_split = img_split[7600:]
print(len(img_test_split))

#moving the images from image folder to train folder
for img2 in img_train_split:
    n2=load_img(img2)
    base2 = os.path.basename(img2)
    fname2 = os.path.splitext(base2)[0]
    train_fname.append(fname2)
    src2 = "/home/itadmin/c:temp/training/image/"+ fname2 + ".jpg"
    dst2 ="/home/itadmin/c:temp/training/image/train"
    shutil.copy2(src2,dst2)
    n2.close()

#moving the images from image folder to validate folder
fname_test=[]
for img5 in img_test_split:
    n5=load_img(img5)
    base5 = os.path.basename(img5)
    fname5 = os.path.splitext(base5)[0]
    fname_test.append(fname5)
    src5 = "/home/itadmin/c:temp/training/image/"+fname5+".jpg"
    dst5 ="/home/itadmin/c:temp/training/image/test"
    shutil.copy2(src5,dst5)
    n5.close()


#loading the profile csv and creating male and female dataframes
df_profile =pd.read_csv("/home/itadmin/c:temp/training/profile/profile.csv")
df_female = df_profile[df_profile.gender == 1]
df_male = df_profile[df_profile.gender ==0]

#separating the images in train folder into female and male
for img in img_train_split:
    n=load_img(img)
    img_train.append(n)
    base = os.path.basename(img)
    fname = os.path.splitext(base)[0]
    idlist_train.append(fname)
    src = "/home/itadmin/c:temp/training/image/train/"+fname+".jpg"
    dst1 ="/home/itadmin/c:temp/training/image/train/male"
    dst2="/home/itadmin/c:temp/training/image/train/female"
    if (fname in df_male.userid.values):
        shutil.copy2(src,dst1)
    elif (fname in df_female.userid.values):
        shutil.copy2(src,dst2)
    n.close()

#separating the images in test folder into female and male
for img1 in img_test_split:
    n1=load_img(img1)
    img_test.append(n1)
    base1 = os.path.basename(img1)
    fname1 = os.path.splitext(base1)[0]
    idlist_test.append(fname1)
    src1 = "/home/itadmin/c:temp/training/image/test/"+fname1+".jpg"
    dst11 ="/home/itadmin/c:temp/training/image/test/male"
    dst22="/home/itadmin/c:temp/training/image/test/female"
    if (fname1 in df_male.userid.values):
        shutil.copy2(src1,dst11)
    elif (fname1 in df_female.userid.values):
        shutil.copy2(src1,dst22)
    n.close()

"""
Program for gender classification using image as input source
"""

#delcaring the image width and height
img_w, img_h = 150, 150
#delcaring tthe path for original vgg16_weights file
top_model_weights_path = '/home/itadmin/Downloads/vgg16_weights.h5'
#declaring the path for training images and validation images
train_data_dir = '/home/c:temp/training/image/train'
validation_data_dir = '/home/c:temp/training/image/test'
#declaring the size of train and validation sample and the batch size for training and validation
nb_train_samples = 7600
nb_validation_samples = 1900
batch_size = 20

#function to extract the bottleneck features 
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(train_data_dir,target_size=(img_w, img_h),batch_size=batch_size,class_mode=None,shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, round(nb_train_samples // batch_size))
    np.save(open('bottleneck_features_train2', 'wb'),bottleneck_features_train)
    generator = datagen.flow_from_directory(validation_data_dir,target_size=(img_w, img_h),batch_size=batch_size,class_mode=None,shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, round(nb_validation_samples // batch_size))
    np.save(open('bottleneck_features_validation2', 'wb'),bottleneck_features_validation)


#call to the above function
save_bottlebeck_features()

#labeling the extracted bottlencek features
epochs=10
train_data = np.load(open('bottleneck_features_train2','rb'))
train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
validation_data = np.load(open('bottleneck_features_validation2','rb'))
validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

#fitting our last two fully  connected layers of our dataset on vgg16 
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels,epochs=epochs,batch_size=batch_size,validation_data=(validation_data, validation_labels))

#saving the weights file after the bottlencek feature extraction process    
model_json = model.to_json()
with open('model_def_gen_v1.json','w') as json_file:
    json_file.write(model_json)
model.save_weights(top_model_weights_path)



#loading the vgg16 model from keras
base_model = applications.VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
batch_size = 16

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_h, img_w),batch_size=batch_size,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size=(img_h, img_w),batch_size=batch_size,class_mode='binary')

# fine-tune the model
model.fit_generator(train_generator,steps_per_epoch=nb_train_samples // batch_size,epochs=epochs,validation_data=validation_generator,validation_steps=nb_validation_samples // batch_size)
model_json = model.to_json()
#saving the model and the wieghts file
with open('model_def_vgg1.json','w') as json_file:
    json_file.write(model_json)
model.save(top_model_weights_path)
model.save_weights(top_model_weights_path)


