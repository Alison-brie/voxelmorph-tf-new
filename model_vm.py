import tensorflow as tf
import neuron.layers as nrn_layers
from keras.layers import UpSampling3D

class conv_block(object):
    def __init__(self, name='conv_block'):
        self.name = name

    def __call__(self, x_in, nf, strides=1):
        conv = tf.layers.Conv3D(nf, (3, 3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x_in)
        x_out = tf.nn.leaky_relu(conv, 0.2)
        return x_out

class UNet(object):
    def __init__(self, name='vm'):
        self.name = name
        self.enc_nf = [16, 32, 32, 32]
        self.dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.conv_block = conv_block()

    def __call__(self, images_0, images_1, reuse=False):# reuse=False
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            x_in = tf.concat([images_1, images_0], axis=-1)

            # down-sample path (encoder)
            x_enc = [x_in]
            for i in range(len(self.enc_nf)):
                x_enc.append(self.conv_block(x_enc[-1], self.enc_nf[i], 2))

            # up-sample path (decoder)
            x = self.conv_block(x_enc[-1], self.dec_nf[0])
            x = UpSampling3D()(x)
            x = tf.concat([x, x_enc[-2]], axis=-1)
            x = self.conv_block(x, self.dec_nf[1])
            x = UpSampling3D()(x)
            x = tf.concat([x, x_enc[-3]], axis=-1)
            x = self.conv_block(x, self.dec_nf[2])
            x = UpSampling3D()(x)
            x = tf.concat([x, x_enc[-4]], axis=-1)
            x = self.conv_block(x, self.dec_nf[3])
            x = self.conv_block(x, self.dec_nf[4])

            # only upsampleto full dim if full_size
            # here we explore architectures where we essentially work with flow fields
            # that are 1/2 size

            x = UpSampling3D()(x)
            x = tf.concat([x, x_enc[0]], axis=-1)
            x = self.conv_block(x, self.dec_nf[5])

            # optional convolution at output resolution (used in voxelmorph-2)
            if len(self.dec_nf) == 7:
                x = self.conv_block(x, self.dec_nf[6])

            # transform the results into a flow field.
            # flow = tf.layers.Conv3D(3, kernel_size=3, padding='same', name='flow',
                                    # kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
            flow = tf.layers.Conv3D(3, (3, 3, 3), (1, 1, 1), 'same', kernel_initializer='he_normal')(x)

            # warp the source with the flow
            y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([images_1, flow])

            return flow, y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]