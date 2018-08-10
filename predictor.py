import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


#Fetch the classifier using keras
classifier = load_model('classifier.h5')

#Fetch the training images from its folder
#Train the Classifier
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

test_image = image.load_img('cat1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices


if result[0][0] == 1:
	print("It is a dog")
else:
	print("It is a cat")