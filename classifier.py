#Import Keras Library and Packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#We create a Sequential Class
classifier = Sequential()

#Add the Convolution Step
classifier.add(Conv2D(32, (3, 3), input_shape  = (64, 64, 3), activation='relu'))

#Add the pooling layer to help reduce the size of the images
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten to convert to a 1-dimensional vector
classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compile The classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Train the Classifier
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

#Test the Classifier
test_datagen = ImageDataGenerator(rescale = 1./255)

#Fetch the training images from its folder
training_set = train_datagen.flow_from_directory('training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

#Fetch the test images from directory
test_set = test_datagen.flow_from_directory('test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

#Fix the data
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 2, validation_data = test_set, validation_steps = 2000)

classifier.save('classifier.h5')