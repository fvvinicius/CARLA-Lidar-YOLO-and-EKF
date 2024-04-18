# %%
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, \
                                    Dense, Dropout, MaxPooling2D, Input, MaxPool2D
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow_datasets import ImageFolder

import numpy as np

# %% [markdown]
# ### Preprocessing

# %%
def resize_image(images: np.ndarray, new_shape: Tuple[int, int]):
    """Resize the image to the input shape of the TinyResNet."""
    images = tf.cast(images, tf.float32)
    images = tf.math.subtract(tf.math.divide(images, 127.5), 1.0)
    images = tf.image.resize(images, size=new_shape)
    return images


def augment_image(images: np.ndarray, labels: np.ndarray):
    """Augment the image to generate more pictures."""
    images = tf.image.random_brightness(images, max_delta=0.5)
    images = tf.image.random_flip_left_right(images)
    # image = tf.image.random_flip_up_down(image)
    return images, labels

# %% [markdown]
# ### Class that contains dataset loading and model training

# %%
class TldTrainingSession:
    """Representing a session for training the traffic light detection model."""

    def __init__(self):
        self.model: Model = None
        self.optimizer: Optimizer = Adam(learning_rate=0.001, amsgrad=True)
        self.loss_func: Loss = SparseCategoricalCrossentropy()
        self.input_shape: [None, 32, 32, 3]
        self.batch_size: int = 512
        self.class_dict: {0: 'backside', 1: 'green',2: 'red', 3: 'yellow'}
        self.weights_path: str = ''
        self.images_path: str = './traffic_light_data'
        self.log_dir: str = './logs'
            
        if self.model is None:
            self.model = TldTrainingSession._create_model(4)
            self.model.build([None, 32, 32, 3])

        print(self.model.summary())
        self.model.compile(self.optimizer, loss=self.loss_func, metrics=['accuracy'])

        if self.weights_path:
            self.model.load_weights(self.weights_path)

    def run_training(self, method : str = 'one'):
        """Train the model and save the weights afterwards."""
        
        ds_train, ds_val = self._load_datasets() # 32 images pro batch
        
        train_callbacks = [TensorBoard(log_dir=self.log_dir)]
        self.model.fit(x=ds_train, validation_data=ds_val,
                       epochs=10, steps_per_epoch=500,
                       callbacks=train_callbacks)

        print('END OF TRAINING')
        self.model.save('traffic_light_model.h5')
        
        print('Model configuration : ')
        loaded_model = tf.keras.models.load_model("traffic_light_model.h5")

        evaluation = loaded_model.evaluate(x=ds_val)
        print("Evaluation loss and accuracy: ", evaluation)

        if method == 'one':
            chosen_image = self.load_one_image()
            chosen_image = np.expand_dims(chosen_image, axis=0)
            a = loaded_model.predict(chosen_image)
            print('Predicted class : ', np.argmax(a))
            
            
        elif method == 'batch':
            predicted_batch = loaded_model.predict(ds_val)
            print("Batch prediction : ", predicted_batch)
            for image_batch, label_batch in ds_val:
                pred = np.expand_dims(image_batch[0], axis=0)
                a = loaded_model.predict(pred)
                print('Predicted class : ', np.argmax(a))
                print('True class : ', label_batch[0])
                break
    
        print('End of TL classification')

    def _create_model(num_classes: int) -> Model:
        """Create a convolution neural network."""
        my_input = Input(shape=(32,32, 3))
    
        x = Conv2D(32, (3,3), activation ='relu')(my_input)  # hyper parameters; change to alter accuracy
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
            
        x = Conv2D(128, (3,3), activation ='relu')(x) # hyper parameters; change to alter accuracy
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)  # Looks at data and does normalization
        
        x = Flatten()(x)
        # x = GlobalAvgPool2D()(x)      # Takes output from batch norm and computes average
        x = Dense(128, activation='relu')(x)   
        x = Dense(num_classes, activation='softmax')(x)     # Output layer form network, 10 possible values, softmax reflects a probability

        

        return Model(inputs=my_input, outputs=x)    
        
        # return Sequential([
        #     Conv2D(filters=4, kernel_size=[5, 5], padding='same', activation='relu'),
        #     BatchNormalization(),
        #     Conv2D(filters=4, kernel_size=[5, 5], padding='same', activation='relu'),
        #     MaxPooling2D(),
        #     Conv2D(filters=4, kernel_size=[3, 3], padding='same', activation='relu'),
        #     MaxPooling2D(),
        #     Conv2D(filters=4, kernel_size=[3, 3], padding='same', activation='relu'),
        #     MaxPooling2D(),
        #     Flatten(),
        #     Dropout(rate=0.1),
        #     Dense(num_classes, activation='softmax')
        # ])

    def _load_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare the dataset for training and validation."""
        builder = ImageFolder(self.images_path)
        ds_train: tf.data.Dataset = builder.as_dataset(split='train', as_supervised=True)
        ds_val: tf.data.Dataset = builder.as_dataset(split='val', as_supervised=True)

        resize_shape = (32, 32)
        resize_op = lambda x, y: (resize_image(x, resize_shape), y)

        ds_train = ds_train.map(resize_op).map(augment_image)
        ds_train = ds_train.shuffle(buffer_size=builder.info.splits['train'].num_examples).repeat().batch(self.batch_size)

        # ds_train = ds_train.shuffle(buffer_size=50).repeat().batch(self.batch_size)
        ds_val = ds_val.map(resize_op).batch(self.batch_size)
        return ds_train, ds_val

    def load_one_image(self) -> np.ndarray:
        """loads one image from a file"""
        path = "./traffic_light_data/train/2/red_1.jpg"
        loaded_image = tf.keras.preprocessing.image.load_img(path)
        loaded_image.show()
        image_to_predict =  np.asarray(loaded_image)
        print("Shape of the chosen image : ", image_to_predict.shape)
        
        #resizing
        resized_image = resize_image(image_to_predict, (32, 32))
        print('New shape of resized_image', resized_image.shape)

        return resized_image

# %%
session = TldTrainingSession()
# one to predict any selected images (chosen in load_one_image method) or batch of tested values
session.run_training('batch') # one or batch

# %%



