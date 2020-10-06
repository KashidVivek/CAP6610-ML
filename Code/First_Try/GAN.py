import tensorflow as tf 
import numpy as np 
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

latent_dim = 28
HEIGHT = 28
WIDTH = 28
channels = 1
mnist = tf.keras.datasets.mnist

gen_input = tf.keras.Input(shape=(latent_dim))
x = tf.keras.layers.Dense(128*14*14)(gen_input)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Reshape((14,14,128))(x)

x = tf.keras.layers.Conv2D(256,5,padding='same')(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2DTranspose(256,4,strides=2,padding='same')(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(256,5,padding='same')(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(256,5,padding='same')(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(channels,7,activation='softmax',padding='same')(x)
generator = tf.keras.models.Model(gen_input,x)
# generator.summary()

discriminator_input = tf.keras.Input(shape=(HEIGHT,WIDTH,channels))
x = tf.keras.layers.Conv2D(128, 3)(discriminator_input)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(128, 4, strides=2)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(128, 4, strides=2)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(128, 4, strides=2)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

discriminator = tf.keras.models.Model(discriminator_input, x)
# discriminator.summary()

discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=0.0008,clipvalue=1.0,decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')

discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)
gan_optimizer = tf.keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train[y_train.flatten() == 4]

x_train = x_train.reshape((x_train.shape[0],) +(HEIGHT, WIDTH, channels)).astype('float32') / 255.

x_train = x_train / 255.0

iters = 1000
batch_size = 20
save_dir = '/home/vivek/genData'

start = 0
for step in range(iters):
	random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))

	generated_images = generator.predict(random_latent_vectors)

	stop = start + batch_size
	real_images = x_train[start:stop]

	combined_images = np.concatenate([generated_images,real_images])

	labels = np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])
	labels += 0.05 * np.random.random(labels.shape)

	d_loss = discriminator.train_on_batch(combined_images,labels)

	random_latent_vectors = np.random.normal(size=(batch_size,latent_dim)) 
	misleading_targets = np.zeros((batch_size,1))
	a_loss = gan.train_on_batch(random_latent_vectors,misleading_targets)

	start += batch_size
	if start > len(x_train) - batch_size:
		start = 0
	print('Discriminator Loss:',d_loss)
	print('Generator Loss:',a_loss)
	print('--------------------')
	if step % 100 == 0:
		gan.save_weights('gan.h5')

		img = image.array_to_img(generated_images[0]*255.0,scale=False)
		img.save(os.path.join(save_dir,'generated_img'+str(step)+'.png'))

		img = image.array_to_img(real_images[0] * 255., scale=False)
		img.save(os.path.join(save_dir,'real_img' + str(step) + '.png'))
