import numpy as np
from getdata import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K
K.set_image_dim_ordering('th')


x_train, x_test, y_train, y_test = load()

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3 , 100, 100)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.load_weights("weights.hdf5")

model.summary()

out = model.predict_proba(x_test)
out = np.array(out)
out

threshold = np.arange(0.1,0.9,0.1)

acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(y_test[:,i],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.max()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []
	
best_threshold

y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

y_pred  #predicted labels

y_test #actual labels
hamming_loss(y_test,y_pred)  #the loss should be as low as possible and  the range is from 0 to 1
total_correctly_predicted = len([i for i in range(len(y_test)) if (t[i]==y_pred[i]).sum() == 5])
total_correctly_predicted/400. #exact accuracy for eg y_pred = [0,0,1,1,1] and y_test=[0,0,1,1,1]

total_correctly_predicted

from IPython.display import Image

Image(filename='test_image.jpg')

import cv2

img = cv2.imread("test_image.jpg")
img.shape
img = cv2.resize(img,(100,100))
img.shape
img = img.transpose((2,0,1))
img.shape

img = img.astype('float32')
img = img/255
img = np.expand_dims(img,axis=0)
img.shape
pred = model.predict(img)
pred
y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])
y_pred
classes = ['desert','mountains','sea','sunset','trees']
[classes[i] for i in range(5) if y_pred[i]==1 ]  #extracting actual class name