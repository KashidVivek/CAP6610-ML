import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

HEIGHT = 28
WIDTH = 28
LATENT_DIM = 100
CHANNELS = 1

def define_discriminator():
	model = Sequential(name='DIS')
	# model.add(Dense(64,activation='relu'))
	# model.add(Dense(128,activation='relu'))
	# model.add(Dropout(0.4))
	
	model.add(Conv2D(64,3,strides=(2,2),padding='same',input_shape=(28,28,1)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))

	model.add(Conv2D(64,3,strides=(2,2),padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
 
	# model.add(Conv2D(64,3,strides=(2,2),padding='same'))
	# model.add(LeakyReLU(alpha=0.2))
	# model.add(Dropout(0.4))

	model.add(Flatten())
	model.add(Dense(1,activation='sigmoid'))

	opt = tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.5)
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
	# print(model.summary())
	return model

def define_generator():
	model = Sequential(name='GEN')
	model.add(Dense(64*7*7,input_dim=LATENT_DIM))
	model.add(LeakyReLU())
 
	model.add(Dense(128*7*7))
	model.add(LeakyReLU())

	model.add(Reshape((7,7,128)))

	model.add(Conv2DTranspose(128,4,strides=(2,2),padding='same'))
	model.add(LeakyReLU())

	model.add(Conv2DTranspose(128,4,strides=(2,2),padding='same'))
	model.add(LeakyReLU())
	
	# model.add(Conv2D(64,(7,7),activation='sigmoid',padding='same'))
	model.add(Conv2D(CHANNELS,(7,7),activation='sigmoid',padding='same'))
	# print(model.summary())
	return model


def define_GAN(generator,discriminator):
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	opt = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5)
	model.compile(loss='binary_crossentropy',optimizer=opt)
	return model


def load_dataset():
	(real_images,real_labels),(_,_) = mnist.load_data()
	real_images = real_images[real_labels.flatten() == 6]
	real_images = real_images.reshape((real_images.shape[0],)+(HEIGHT,WIDTH,CHANNELS)).astype('float32') / 255.
	return real_images

latent_dim = 100
real_images = load_dataset()
generator = define_generator()
discriminator = define_discriminator()
GAN = define_GAN(generator,discriminator)
epochs = 20
batch_size = 128
batch_per_epoch = int(real_images.shape[0] / batch_size)
save_dir = '/home/vivek/genData'
images = []
half_batch = batch_size // 2


for i in range(epochs):
	for j in range(batch_per_epoch):
		# get real images
		real_batch_index = tf.random.uniform(shape=(batch_size,1),minval=0,maxval=real_images.shape[0],dtype=tf.int32)
		real_batch_index = tf.reshape(real_batch_index,(batch_size))
		real_images_batch = real_images[real_batch_index]
		real_labels = tf.ones([batch_size,1])

		#get fake images
		random_vectors = tf.random.normal((batch_size,LATENT_DIM))
		generated_images = generator.predict(random_vectors)
		fake_labels = tf.zeros([batch_size,1])
		
		X_train, y_train = tf.concat([real_images_batch,generated_images],0), tf.concat([real_labels,fake_labels],0)
		discriminator_loss,_ = discriminator.train_on_batch(X_train,y_train)

		random_vectors_gan = tf.random.normal((batch_size*2,LATENT_DIM))
		labels_gan = tf.ones([batch_size*2,1])
		gan_loss = GAN.train_on_batch(random_vectors_gan,labels_gan)
		print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, discriminator_loss, gan_loss))
		
	if i % 2 == 0:
		(_,acc_real) = discriminator.evaluate(real_images_batch,real_labels)
		(_,acc_fake) = discriminator.evaluate(generated_images,fake_labels)
		print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
		images.append(generated_images)

