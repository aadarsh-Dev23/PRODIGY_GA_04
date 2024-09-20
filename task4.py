import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the dataset (example using CIFAR-10)
def load_data():
    (train_images, _), (test_images, _) = keras.datasets.cifar10.load_data()
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    train_images = tf.image.resize(train_images, [256, 256])
    return train_images

# Build the generator model
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
    return keras.Model(inputs, x)

# Build the discriminator model
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 6))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, x)

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the pix2pix model
class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def call(self, inputs):
        generated_image = self.generator(inputs[0])
        combined_input = tf.concat([inputs[0], generated_image], axis=-1)
        validity = self.discriminator(combined_input)
        return generated_image, validity

pix2pix_model = Pix2Pix(generator, discriminator)

# Prepare dataset
def prepare_dataset(images, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset

# Define loss functions
def generator_loss(generated_image, target):
    return tf.reduce_mean(tf.losses.mean_squared_error(target, generated_image))

def discriminator_loss(real_validity, fake_validity):
    return tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(real_validity), real_validity)) + \
           tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(fake_validity), fake_validity))

# Training function
def train(dataset, epochs=100):
    for epoch in range(epochs):
        for real_images in dataset:
            target_images = real_images  # Modify this for your dataset
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(real_images)
                combined_input = tf.concat([real_images, generated_images], axis=-1)

                real_validity = discriminator(combined_input)
                fake_validity = discriminator(tf.concat([real_images, target_images], axis=-1))

                gen_loss = generator_loss(generated_images, target_images)
                disc_loss = discriminator_loss(real_validity, fake_validity)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        print(f'Epoch {epoch + 1}/{epochs} completed.')

# Visualize results
def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Generated Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Rescale for visualization
        plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    train_images = load_data()
    dataset = prepare_dataset(train_images)

    # Train the model
    train(dataset, epochs=50)

    # Test the model and visualize results
    test_input = train_images[:1]  # Use first image as an example
    generate_images(generator, test_input)
