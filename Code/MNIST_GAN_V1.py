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
	# model.add(Dense(64*7*7,input_dim=LATENT_DIM))
	# model.add(LeakyReLU())
 
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
	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
	return model


def load_dataset():
	(real_images,real_labels),(_,_) = mnist.load_data()
	# real_images = real_images[real_labels.flatten() == 6]
	real_images = real_images.reshape((real_images.shape[0],)+(HEIGHT,WIDTH,CHANNELS)).astype('float32') / 255.
	return real_images

def save_plots(generated_images,k):
	fig=plt.figure(figsize=(5, 5))
	for i in range(0, 5*5):
			image = tf.reshape(generated_images[i],(28,28))
			fig.add_subplot(5, 5, 1+i)
			plt.axis('off')
			plt.imshow(image,cmap='gray')
	# plt.imsave(f"epoch_{k}_GAN.png",image)
	plt.savefig(f"epoch_{k}_GAN.png")
	plt.close()

def draw_loss_plots(dl,gl):
	title = "Losses Compared"
	epochs = range(1, len(dl)+1)
	plt.plot(epochs,dl, '-b',marker='o', label='Discriminator loss')
	plt.plot(epochs,gl, '-r', marker='o',label='GAN Loss')
	plt.xticks(np.arange(0,len(epochs)+1,5))
	plt.xlabel("Epochs")
	plt.legend(loc='upper right')
	plt.title(title)

	# save image
	plt.savefig(title+".png")
	plt.close()


def draw_discriminator_accuracy(real,fake):
	title = "Real vs Fake Accuracy"
	epochs = range(1, len(real)+1)
	plt.plot(epochs,real, '-b',marker='o', label='Accuracy on Real image')
	plt.plot(epochs,fake, '-r', marker='o',label='Accuracy on Fake image')
	plt.xticks(np.arange(0,len(epochs)+1,5))
	plt.xlabel("Epochs")
	plt.legend(loc='upper right')
	plt.title(title)

	# save image
	plt.savefig(title+".png")
	plt.close()



latent_dim = 100
real_images = load_dataset()
generator = define_generator()
discriminator = define_discriminator()
GAN = define_GAN(generator,discriminator)
epochs = 100
batch_size = 128
batch_per_epoch = int(real_images.shape[0] / batch_size)
save_dir = '/home/vivek/genData'
images = []
half_batch = batch_size // 2
discriminator_losses = []
GAN_losses = []
discriminator_real_images_accuracy = []
discriminator_fake_images_accuracy = []


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
		gan_loss,_ = GAN.train_on_batch(random_vectors_gan,labels_gan)
		print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, discriminator_loss, gan_loss))

	if i == 0:
		save_plots(generated_images,i)
	
	if i % 1 == 0:
		(_,acc_real) = discriminator.evaluate(real_images_batch,real_labels)
		(_,acc_fake) = discriminator.evaluate(generated_images,fake_labels)
		print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
		discriminator_real_images_accuracy.append(acc_real*100)
		discriminator_fake_images_accuracy.append(acc_fake*100)
		# save_plots(generated_images)
		images.append(generated_images)

	if i % 10 == 0:
		save_plots(generated_images,i)
	
	discriminator_losses.append(discriminator_loss)
	GAN_losses.append(gan_loss)
draw_loss_plots(discriminator_losses,GAN_losses)
draw_discriminator_accuracy(discriminator_real_images_accuracy,discriminator_fake_images_accuracy)