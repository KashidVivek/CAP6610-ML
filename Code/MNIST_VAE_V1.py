import tensorflow as tf
from tensorflow.keras.layers import Flatten, Lambda, Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

latent_dim = 16
channels = 1
height = width = 28

def sampling(mean_stddev_layer):
  mean, std_dev = mean_stddev_layer
  sample = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean),mean=0.0,stddev=1.0)
  random_sample = mean + tf.keras.backend.exp(std_dev/2) * sample
  return random_sample

def encoder():
  input_layer = tf.keras.layers.Input(shape=(28,28,1))
  x = Conv2D(filters=1,kernel_size=3,strides=2,padding='same')(input_layer)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = Conv2D(filters=32,kernel_size=3,strides=2,padding='same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)

  # x = Conv2D(filters=64,kernel_size=3,strides=2,padding='same')(x)
  # x = BatchNormalization()(x)
  # x = LeakyReLU(alpha=0.2)(x)

  x = Flatten()(x)
  x = Dense(32,activation='relu')(x)
  # x = LeakyReLU(alpha=0.2)(x)

  x_mean = Dense(latent_dim)(x)
  x_stddev = Dense(latent_dim)(x)
  output_layer = Lambda(sampling,name='encoder_output')([x_mean,x_stddev])
  encoder = Model(input_layer,output_layer,name='encoder')
  print(encoder.summary())
  return encoder,x_mean,x_stddev

def decoder():
  input_layer = tf.keras.layers.Input(shape=(latent_dim))
  x = Dense(7*7*64)(input_layer)

  x = Reshape((7,7,64))(x)

  x = Conv2DTranspose(filters=64,kernel_size=(3,3),padding='same',strides=2)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)

  x = Conv2DTranspose(filters=64,kernel_size=(3,3),padding='same',strides=2)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)

  x = Conv2DTranspose(filters=channels,kernel_size=(3,3),padding='same',strides=1)(x)
  x = LeakyReLU()(x)

  decoder = Model(input_layer,x,name='decoder')
  print(decoder.summary())
  return decoder

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss
def VAE(encoder,decoder,x_mean,x_stddev):
  input_layer = tf.keras.layers.Input(shape=(28,28,1))
  x = encoder(input_layer)
  output_layer = decoder(x)
  vae = Model(input_layer,output_layer,name='VAE')
  opt = tf.keras.optimizers.Adam(lr=0.0002)
  vae.compile(optimizer=opt,loss=loss_func(x_mean,x_stddev))
  return vae

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0

x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)) 
x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))

encoder_model,x_mean,x_stddev = encoder()
decoder_model = decoder()
vae = VAE(encoder_model,decoder_model,x_mean,x_stddev)

vae.fit(x_train, x_train, epochs=20, batch_size=32, shuffle=True, validation_data=(x_test, x_test))
encoded_data = encoder_model.predict(x_test)
decoded_data = decoder_model.predict(encoded_data)