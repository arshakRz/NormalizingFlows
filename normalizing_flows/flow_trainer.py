import tensorflow as tf
from tensorflow import keras as tfk
import numpy as np
from time import time

def train(model, X, steps = 500):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    -tf.reduce_mean(model.flow.log_prob(X)) 
    @tf.function #Adding the tf.function makes it about 10-50 times faster!!!
    def train_step(X): 
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(model.flow.log_prob(X)) 
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss      

    start = time()
    for i in range(steps):
        loss = train_step(X)
        if (i % 100 == 0):
            print("Epoch:",i, ", Loss:",loss.numpy(), ", Time:",(time()-start))
            start = time()
    return model

