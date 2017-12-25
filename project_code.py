#Create the convolutional neural network that will discern between cars
#and objects that are not cars. The architecture will contain 9 convolutional
#layers and three fully connected layers.

#First, obtain all of the training vehicle images
images = []
labels = []
import os
import cv2
#get a list of all the directories inside vehicles directory, then non-vehicles
for index,img_type in enumerate(['non-vehicles','vehicles']):
    lod = [x[0] for x in os.walk('C:/Users/mdf30/src/CarND-Vehicle-Detection/'+img_type)]
    #for each directory in the list of directories
    for directory in lod[::-1]:
        #for each file in the directory, if it ends in .png, it is an image.
        #append to the list of images
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                images.append(cv2.imread(directory+'/'+filename))
                labels.append(index)
                break
        break
    break


#Next, train all of the images using a convolutional neural network
model = Sequential()
model.add(Lambda(lambda x: (x -128.0)/128.0, input_shape=(64,64,3)))
#model.add(Cropping2D(cropping=((35,15), (0,0))))
model.add(Convolution2D(24,5,5,activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36,5,5,activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(48,5,5,activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64,3,3,activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64,3,3,activation="relu", border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#Compile and train the model
model.compile(loss='binary_crossentropy',optimizer='adam')
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=4)

#Save the model to see it run on local machine
model.save('model_nvidia4.h5')

#plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
    


