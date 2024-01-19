import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns



(X_train,y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


print('Before normalization: Min={}, Max={}'.format(X_train.min(),X_train.max()))

X_train_max = X_train.max()

X_train = X_train/X_train_max
X_test = X_test/X_train_max

print('Before normalization: Min={}, Max={}'.format(X_train.min(),X_train.max()))


# Display one image for each digit
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 4))
for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    ax[digit].imshow(255 - X_train[digit_indices[0]], cmap='gray')
    ax[digit].set_title(f'Digit {digit}')
    ax[digit].axis('off')

plt.tight_layout()
#plt.show()


hidden1 = 150




model = keras.models.Sequential()

model.add( keras.layers.Input((28,28,1)) )

model.add( keras.layers.Conv2D(8, (3,3), activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)) )
model.add( keras.layers.Dropout(0.2) ) 


model.add( keras.layers.Conv2D(16, (3,3), activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)) )
model.add( keras.layers.Dropout(0.2) ) 


model.add( keras.layers.Flatten() )
model.add( keras.layers.Dense(hidden1, activation='relu') )
model.add( keras.layers.Dropout(0.5) ) 

model.add( keras.layers.Dense(10, activation='softmax') )


model.summary()


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


batch_size = 512
epochs = 16

history = model.fit(X_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (X_test,y_test))

print(history)

score = model.evaluate(X_test, y_test, verbose = 0)

print(f'Test loss : {score[0]:4.4f}')

print(f'Test accuracy : {score[1]:4.4f}')

# Retrieving training and validation metrics
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Visualizing the evolution of accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Visualizing the evolution of loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Making predictions on the test set
y_predict = model.predict(X_test)
y_pred = np.argmax(y_predict, axis=-1)


# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()