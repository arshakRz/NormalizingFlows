from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
import tensorflow as tf

class IAF(tf.keras.models.Model):

    def __init__(self, output_dim, num_bijectors, hidden = 32, batch_norm = 0, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.nets=[]
        self.bijectors=[] 
        for i in range(num_bijectors):
            iaf=tfb.Invert(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
            params=2, hidden_units=[hidden, hidden])))
            self.bijectors.append(iaf)
            if batch_norm == 1 and i % 3 == 0:
                self.bijectors.append(tfb.BatchNormalization(name='batch_norm%d' % i))
            self.bijectors.append(tfb.Permute([1,0]))
        bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.), sample_shape=[output_dim]), 
            bijector = bijector) 