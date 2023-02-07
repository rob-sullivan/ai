#CNN
#import libraries
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator #keras.io/api/preprocessing/image/

#tf.__version__ # should be 2.20

class CNN():

    def __init__(self, train_path, test_path):
        #ai hyper parameters

        #preprocess params
        #image data generator parameters
        self.rescale = 1./255 #feature scaling
        self.shear_range = 0.2
        self.zoom_range = 0.2
        self.horizontal_flip = True

        #folder locations
        self.training_set_path = train_path
        self.test_set_path = test_path

        #build params
        #image classification
        self.target_size = (64, 64)
        self.batch_size = 32
        self.class_mode = 'binary'

        #convolution layers
        self.activation = ['relu', 'sigmoid']
        self.filters = 32
        self.kernel_size = 3
        self.input_shape = [64, 64, 3]

        # max pooling
        self.pool_size = 2
        self.strides = 2

        # full connection, output layer
        self.units = ['128','1']

        #train params
        self.epochs = 25

        #running the ai
        self.preprocessData()
        self.buildAnn()
        self.trainAnn()

    def preprocessData(self):
        #data preprocessing
        #preprocessing training set
        train_datagen = ImageDataGenerator(
            rescale = self.rescale,
            shear_range = self.shear_range,
            zoom_range = self.zoom_range,
            horizontal_flip = self.horizontal_flip
        )
        self.training_set = train_datagen.flow_from_directory(
            self.training_set_path,
            target_size = self.target_size,
            batch_size = self.batch_size,
            class_mode = self.class_mode
        )

        #preprocessing test set
        test_datagen = ImageDataGenerator(rescale = 1./255)
        self.test_set = test_datagen.flow_from_directory(
            self.test_set_path,
            target_size = self.target_size,
            batch_size = self.batch_size,
            class_mode = self.class_mode
)

    def buildAnn(self):
        #initialise the cnn
        self.cnn = tf.keras.models.Sequential()

        #1 convolution
        self.cnn.add(tf.keras.layers.Conv2D(
            filters = self.filters,
            kernel_size = self.kernel_size,
            activation = self.activation[0],
            input_shape = self.input_shape
        ))

        #2 pooling (max pooling)
        self.cnn.add(tf.keras.layers.MaxPool2D(
            pool_size = self.pool_size ,
            strides = self.strides,
        ))

        #add second convolution layer
        self.cnn.add(tf.keras.layers.Conv2D(
            filters = self.filters,
            kernel_size = self.kernel_size,
            activation = self.activation[0],
        ))

        self.cnn.add(tf.keras.layers.MaxPool2D(
            pool_size = self.pool_size,
            strides = self.strides,
        ))

        #3 flatten
        self.cnn.add(tf.keras.layers.Flatten())

        #4 full connection
        self.cnn.add(tf.keras.layers.Dense(
            units = self.units[0],
            activation = self.activation[0],
        ))
        #5 output layer
        self.cnn.add(tf.keras.layers.Dense(
            units = self.units[1],
            activation = self.activation[1],
        ))

    def trainAnn(self):
        #compile cnn
        self.cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        #train cnn on training set and evaluyate it on test set
        self.cnn.fit(x = self.training_set, validation_data = self.test_set, epochs = self.epochs)
        #ValueError: Unknown loss function:  binary_crossentropy. Please ensure this object is passed to the `custom_objects` argument. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.
        
    def predict(self, data):
        #make a single prediction
        test_image = image.load_img(data, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = self.cnn.predict(test_image/255.0) # 1=dog and 0=cat
        
        print(self.training_set.class_indices)

        if result[0][0] > 0.5:
            prediction = 'dog'
        else:
            prediction = 'cat'
        
        print(prediction)

if __name__ == "__main__":
    #run artificial neural network
    ai = CNN('dataset/training_set','dataset/training_set')
    ai.predict('dataset/single_prediction/cat_or_dog_1.jpg')
    