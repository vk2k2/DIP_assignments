from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy
#load data and reshape, normalize
data=scipy.io.loadmat("sat-6-full.mat")

train_X = data['train_x']
train_x_new = np.zeros((324000,28,28,4),np.uint8)
for i in range(0,324000):
    train_x_new[i] = train_X[:,:,:,i]
train_x_new =train_x_new.astype('float32')/255 #comment out if using too much RAM  (will lower accuraacy)

train_Y = np.transpose(data['train_y'])

test_X = data['test_x']
test_x_new = np.zeros((81000,28,28,4),np.uint8)
for i in range(0,81000):
    test_x_new[i] = test_X[:,:,:,i]
test_x_new =test_x_new.astype('float32')/255 #comment out if using too much RAM (will lower accuracy)

test_Y = np.transpose(data['test_y'])

model = keras.Sequential()
model.add(layers.Conv2D(40, (5,5), activation='relu', input_shape=(28,28,4), padding = 'same'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(40, (3,3), activation='relu', padding = 'same'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', padding = 'same'))
model.add(layers.Flatten())
model.add(layers.Dense(6, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(train_x_new, train_Y, batch_size=324, epochs=7, validation_split=0.1)

print(model.summary())

score = model.evaluate(test_x_new,test_Y)

print(" accuracy obtained is:",score[1])

# Epoch 1/7
# 900/900 [==============================] - 124s 137ms/step - loss: 0.0182 - accuracy: 0.9263 - val_loss: 0.0070 - val_accuracy: 0.9736
# Epoch 2/7
# 900/900 [==============================] - 131s 146ms/step - loss: 0.0060 - accuracy: 0.9765 - val_loss: 0.0059 - val_accuracy: 0.9776
# Epoch 3/7
# 900/900 [==============================] - 122s 135ms/step - loss: 0.0049 - accuracy: 0.9810 - val_loss: 0.0044 - val_accuracy: 0.9830
# Epoch 4/7
# 900/900 [==============================] - 120s 134ms/step - loss: 0.0044 - accuracy: 0.9831 - val_loss: 0.0040 - val_accuracy: 0.9851
# Epoch 5/7
# 900/900 [==============================] - 119s 132ms/step - loss: 0.0042 - accuracy: 0.9840 - val_loss: 0.0041 - val_accuracy: 0.9843
# Epoch 6/7
# 900/900 [==============================] - 125s 139ms/step - loss: 0.0039 - accuracy: 0.9851 - val_loss: 0.0041 - val_accuracy: 0.9844
# Epoch 7/7
# 900/900 [==============================] - 126s 140ms/step - loss: 0.0036 - accuracy: 0.9865 - val_loss: 0.0034 - val_accuracy: 0.9874
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_3 (Conv2D)           (None, 28, 28, 40)        4040      
                                                                 
#  max_pooling2d_3 (MaxPooling  (None, 14, 14, 40)       0         
#  2D)                                                             
                                                                 
#  conv2d_4 (Conv2D)           (None, 14, 14, 40)        14440     
                                                                 
#  max_pooling2d_4 (MaxPooling  (None, 7, 7, 40)         0         
#  2D)                                                             
                                                                 
#  conv2d_5 (Conv2D)           (None, 7, 7, 64)          23104     
                                                                 
#  flatten_1 (Flatten)         (None, 3136)              0         
                                                                 
#  dense_1 (Dense)             (None, 6)                 18822     
                                                                 
# =================================================================
# Total params: 60,406
# Trainable params: 60,406
# Non-trainable params: 0
# _________________________________________________________________
# None
# 2532/2532 [==============================] - 59s 23ms/step - loss: 0.0033 - accuracy: 0.9878
#  accuracy obtained is: 0.9878271818161011