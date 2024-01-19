import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist


# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display one image for each digit
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 4))
for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    ax[digit].imshow(255 - X_train[digit_indices[0]], cmap='gray')
    ax[digit].set_title(f'Digit {digit}')
    ax[digit].axis('off')

plt.tight_layout()
#plt.show()

# Reshape and normalize input data
X_train = X_train/ 255.0
X_test = X_test / 255.0




hidden1 = 100
hidden2 = 100

model = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(hidden1, activation='relu'),
    keras.layers.Dense(hidden2, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 512
epochs = 16

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

# Plot the training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)






print(y_pred[0])


plt.figure(figsize=(20, 4))
plt.imshow(255 - X_test[0], cmap='gray')  
plt.title(f'Digit: {y_test[0]}')
plt.axis('off')
plt.show()

print(model.summary())



