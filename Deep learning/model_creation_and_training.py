import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Model creation
model = models.Sequential()

# Create VGG16 layer
vgg16_model = keras.applications.vgg16.VGG16()

# Delete last layer (Dense 1000)
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

# Freeze layer
for layer in model.layers:
    layer.trainable = False

# Compile model, Using optimizers adam with learning rate = 1e-4, loss= 'categorical_crossentropy', and metric=['accuracy']
model.compile(
    tf.keras.optimizers.Adam(lr=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['acc']
)

# You may want to see sumarization of your model
model.summary()

# Use early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Train model

# Train model
history = model.fit(
    'train_ds',
    validation_data='validation_ds',
    epochs=35,
    steps_per_epoch=100,
    callbacks=callback
)

# Save a Model
model.save('VGG16_epochs_35_lr_e-4.h5')

# Plot Training and validation accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
  
epochs = range(len(acc))
  
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

# Plot Training and validation loss
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()