from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Load a training and validating dataset.
train_batches = ImageDataGenerator().flow_from_directory(
    'training_path',
    target_size=(224, 224),
    batch_size=10
)

valid_batches = ImageDataGenerator().flow_from_directory(
    'validation_path',
    target_size=(224, 224),
    batch_size=10
)

test_batches = ImageDataGenerator().flow_from_directory(
    'testing_path',
    target_size=(224, 224),
    batch_size=10
)
