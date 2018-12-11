import csv
import cv2
import numpy as np

#Import Dataset
lines = []
with open('../Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path_center = line[0]
	source_path_left = line[1]
	source_path_right = line[2]

	filename_center = source_path_center.split('/')[-1]
	filename_left = source_path_left.split('/')[-1]
	filename_right = source_path_right.split('/')[-1]

	current_path_center = '../Data/IMG/' + filename_center
	current_path_left = '../Data/IMG/' + filename_left
	current_path_right = '../Data/IMG/' + filename_right

	image_center = cv2.imread(current_path_center)
	image_left = cv2.imread(current_path_left)
	image_right = cv2.imread(current_path_right)

	images.append(image_center)
	measurement = float(line[3])
	measurements.append(measurement)

#Augument Dataset
augmented_images = []
augmented_measurements = []

for image,measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

#Keras needs Numpy arrays
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

#LeNet
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
