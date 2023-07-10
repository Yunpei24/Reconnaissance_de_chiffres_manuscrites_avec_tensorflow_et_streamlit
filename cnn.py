import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Chargement des données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Prétraitement des données
x_train = x_train / 255.0
x_test = x_test / 255.0

# Construction du modèle CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Sauvegarde du modèle
model.save('./mnist_cnn.h5')