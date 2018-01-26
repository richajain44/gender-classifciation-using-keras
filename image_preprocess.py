"""
Program to separate images into train and validation set
and then to create separate folders in train for female and male having equal number of images in both 
and also cresting same folder structure in the validation set
"""

#importing all the necessary modules
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