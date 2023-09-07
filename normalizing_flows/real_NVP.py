import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from keras import losses as tfkl
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from time import time


class NVP(tf.keras.models.Model):

    def __init__(self, output_dim, num_masked, num_bijectors, hidden,  batch_norm = 0, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.nets=[]
        self.bijectors=[] 
        for i in range(num_bijectors): 
            net = tfb.real_nvp_default_template([hidden, hidden])
            self.bijectors.append(
                tfb.RealNVP(shift_and_log_scale_fn=net, 
                            num_masked=num_masked))
            if batch_norm == 1 and i % 3 == 0:
                self.bijectors.append(tfb.BatchNormalization(name='batch_norm%d' % i))
            self.bijectors.append(tfb.Permute([1,0]))
            self.nets.append(net) 
        bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.), sample_shape=[output_dim]), 
            bijector = bijector) 

