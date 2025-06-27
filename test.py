import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import cv2

from sklearn.neighbors import NearestNeighbors
feature_list=np.array(pickle.load(open('embeddings.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))
model=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False
#APNE MODEL MEI HAMNE EK LAYER ADD KARDI LAST VALI HATAKE HAMARI
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


i=image.load_img('sample/shirt.jpeg',target_size=(224,224))
array_of_image=image.img_to_array(i)
expand_image_array=np.expand_dims(array_of_image,axis=0)
preprocess_image=preprocess_input(expand_image_array)
result=model.predict(preprocess_image).flatten()
normalised_result=result/norm(result)

#distance calculate
neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices=neighbors.kneighbors([normalised_result])
print(indices)
for file in indices[0][1:6]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)