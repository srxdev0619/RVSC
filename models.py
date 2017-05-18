from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape, Permute, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate, add
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization


import tensorflow as tf


def unet(inputs,
         trainable=True):
    
    with tf.name_scope('unet'):
        conv1 = LeakyReLU()(Conv2D(64,(3,3), padding='same', trainable=trainable)(inputs))
        conv1 = LeakyReLU()(Conv2D(64, (3,3), padding='same', trainable=trainable)(conv1))
        down1 = Conv2D(64, (2,2), strides = 2, trainable=trainable)(conv1)


        conv2 = LeakyReLU()(Conv2D(128, (3,3), padding='same', trainable=trainable)(down1))
        conv2 = LeakyReLU()(Conv2D(128, (3,3), padding='same', trainable=trainable)(conv2))
        down2 = Conv2D(128, (2,2), strides = 2, trainable=trainable)(conv2)


        conv3 = LeakyReLU()(Conv2D(256, (3,3), padding='same', trainable=trainable)(down2))
        conv3 = LeakyReLU()(Conv2D(256, (3,3), padding='same', trainable=trainable)(conv3))
        down3 = Conv2D(256, (2,2),  strides = 2, trainable=trainable)(conv3)
        

        conv4 = LeakyReLU()(Conv2D(512, (3,3), padding='same', trainable=trainable)(down3))
        conv4 = LeakyReLU()(Conv2D(512, (3,3), padding='same', trainable=trainable)(conv4))
        down4 = Conv2D(512, (2,2),  strides = 2, trainable=trainable)(conv4)

        conv5 = LeakyReLU()(Conv2D(1024, (3,3), padding='same', trainable=trainable)(down4))
        conv5 = LeakyReLU()(Conv2D(1024, (3,3), padding='same', trainable=trainable)(down4))

        convT1 = Conv2DTranspose(512, (2,2),  strides = 2, trainable=trainable)(conv5)
        convT1Merge = Concatenate()([convT1,conv4])
        convT1Merge = LeakyReLU()(Conv2D(512, (3,3), padding='same',
                                         trainable=trainable)(convT1Merge))
        convT1Merge = LeakyReLU()(Conv2D(512, (3,3), padding='same',
                                         trainable=trainable)(convT1Merge))

        convT2 = Conv2DTranspose(256, (2,2),  strides = 2, trainable=trainable)(convT1Merge)
        convT2Merge = Concatenate()([convT2, conv3])
        convT2Merge = LeakyReLU()(Conv2D(256, (3,3), padding='same',
                                         trainable=trainable)(convT2Merge))
        convT2Merge = LeakyReLU()(Conv2D(256, (3,3), padding='same',
                                         trainable=trainable)(convT2Merge))

        convT3 = Conv2DTranspose(128, (2,2),  strides = 2, trainable=trainable)(convT2Merge)
        convT3Merge = Concatenate()([convT3, conv2])
        convT3Merge = LeakyReLU()(Conv2D(128, (3,3), padding='same',
                                         trainable=trainable)(convT3Merge))
        convT3Merge = LeakyReLU()(Conv2D(128, (3,3), padding='same',
                                         trainable=trainable)(convT3Merge))

        convT4 = Conv2DTranspose(64, (2,2) ,  strides = 2, trainable=trainable)(convT3Merge)
        convT4Merge = Concatenate()([convT4, conv1])
        convT4Merge = LeakyReLU()(Conv2D(64, (3,3), padding='same',
                                         trainable=trainable)(convT4Merge))
        convT4Merge = LeakyReLU()(Conv2D(64, (3,3), padding='same',
                                         trainable=trainable)(convT4Merge))

        output = Conv2D(1, (3,3), padding='same', activation='softmax')(convT4Merge)

        return output



def initial_block(inputs,
                 num_filters=13,
                 filter_rows=3,
                 filter_cols=3,
                 strides=(2,2),
                 trainable=True):
    
    conv = Conv2D(num_filters, (filter_rows, filter_cols),
                  padding='same', strides=strides, trainable=trainable)(inputs)
    
    max_pool = MaxPooling2D()(inputs)

    merged = Concatenate(axis=3)([conv, max_pool])
    return merged




def bottleneck_enc(inputs,
                   out_channels,
                   internal_scale=4,
                   asymmetric=0,
                   dilated=0,
                   downsample=False,
                   dropout_rate=0.1,
                   trainable=True):
    with tf.name_scope('enc_bottleneck'):

        mid_channels = int(out_channels/internal_scale)

        encoder = inputs

        input_stride = 2 if downsample else 1
        
        encoder = Conv2D(mid_channels, (input_stride, input_stride),
                         padding='same', strides=input_stride,
                         use_bias=False, trainable=trainable)(encoder) 

        encoder = BatchNormalization(momentum=0.1, trainable=trainable)(encoder)
        encoder = PReLU(shared_axes=[1,2], trainable=trainable)(encoder)

        if not asymmetric and not dilated:
            encoder = Conv2D(mid_channels, (3,3), padding='same', trainable=trainable)(encoder)
        elif asymmetric:
        
            encoder = Conv2D(mid_channels, (1, asymmetric),
                             padding='same', use_bias='False', trainable=trainable)(encoder)
            encoder = Conv2D(mid_channels, (asymmetric, 1),
                             padding='same', trainable=trainable)(encoder)
        elif dilated:

            encoder = Conv2D(mid_channels, (3,3), dilation_rate=dilated,
                             padding='same', trainable=trainable)(encoder)
        else:
            raise(Exception('Invalid Value for asymmetric or dilation'))
        
        encoder = BatchNormalization(momentum=0.1, trainable=trainable)(encoder)
        encoder = PReLU(shared_axes=[1,2], trainable=trainable)(encoder)

        encoder = Conv2D(out_channels, (1,1), padding='same',
                         use_bias=False, trainable=trainable)(encoder)

        encoder = BatchNormalization(momentum=0.1, trainable=trainable)(encoder)
        encoder = SpatialDropout2D(dropout_rate)(encoder)

        down_branch = inputs

        if downsample:
            down_branch = MaxPooling2D()(down_branch)
            down_branch = Permute((1,3,2))(down_branch)
            pad_feature_maps = out_channels - inputs.get_shape().as_list()[3]
            tb_pad = (0,0)
            lr_pad = (0, pad_feature_maps)
            down_branch = ZeroPadding2D(padding=(tb_pad, lr_pad))(down_branch)
            down_branch = Permute((1,3,2))(down_branch)

        encoder = add([encoder, down_branch])
        encoder = PReLU(shared_axes=[1,2])(encoder)

    return encoder


def bottleneck_dec(inputs,
                   out_channels,
                   internal_scale=4,
                   upsample=False,
                   trainable=True,
                   reverse_module=False):

    with tf.name_scope('dec_bottleneck'):
        mid_channels = int(out_channels/internal_scale)

        input_stride = 2 if upsample else 1

        decoder = Conv2D(mid_channels, (1,1), padding='same',
                         use_bias=False, trainable=trainable)(inputs)
        decoder = BatchNormalization(momentum=0.1, trainable=trainable)(decoder)
        decoder = Activation('relu')(decoder)
        
        if upsample:
            decoder = Conv2DTranspose(mid_channels, (3,3), strides=2,
                                      trainable=trainable, padding='same')(decoder)
        else:
            decoder = Conv2D(mid_channels, (3,3), padding='same', trainable=trainable)(decoder)

        decoder = BatchNormalization(momentum=0.1, trainable=trainable)(decoder)
        decoder = Activation('relu')(decoder)

        decoder = Conv2D(out_channels, (1,1), padding='same',
                         use_bias=False, trainable=trainable)(decoder)

        up_branch = inputs

        if decoder.get_shape()[-1] != out_channels or upsample:
            up_branch = Conv2D(out_channels, (1,1), padding='same',
                               use_bias=False, trainable=trainable)(up_branch)
            up_branch = BatchNormalization(momentum=0.1, trainable=trainable)(up_branch)

            if upsample and reverse_module:
                up_branch = UpSampling2D((2,2))(up_branch)
                
        if not upsample or reverse_module:
            up_branch = BatchNormalization(momentum=0.1, trainable=trainable)(up_branch)
        else:
            return up_branch

        decoder = add([decoder, up_branch])
        decoder = Activation('relu')(decoder)

        return decoder
            

        
def enet_encoder(inputs,
                 dropout_rate=0.01,
                 trainable=True):

    enet_enc = initial_block(inputs, trainable=trainable)
    enet_enc = bottleneck_enc(enet_enc,
                              out_channels=64,
                              downsample=True,
                              dropout_rate=dropout_rate,
                              trainable=trainable)

    for _ in range(4):
        enet_enc = bottleneck_enc(enet_enc,
                                  out_channels=64,
                                  dropout_rate=dropout_rate,
                                  trainable=trainable)

    enet_enc = bottleneck_enc(enet_enc,
                              out_channels=128,
                              downsample=True)

    for _ in range(2):
        enet_enc = bottleneck_enc(enet_enc, 128, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, dilated=2, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, asymmetric=5, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, dilated=4, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, dilated=8, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, asymmetric=5, trainable=trainable)
        enet_enc = bottleneck_enc(enet_enc, 128, dilated=16, trainable=trainable)
    return enet_enc



def enet_decoder(inputs,
                 out_channels,
                 activation='softmax',
                 trainable=True):

    enet_dec = bottleneck_dec(inputs, 64, upsample=True,
                              reverse_module = True, trainable=trainable)

    enet_dec = bottleneck_dec(enet_dec, 64, trainable=trainable)
    enet_dec = bottleneck_dec(enet_dec, 64, trainable=trainable)

    enet_dec = bottleneck_dec(enet_dec, 16, upsample=True,
                              reverse_module=True, trainable=True)

    enet_dec = bottleneck_dec(enet_dec, 16)

    enet_dec = Conv2DTranspose(out_channels, (2,2), strides=2, 
                               padding='same', activation=activation)(enet_dec) 

    return enet_dec



def enet(inputs,
         out_channels,
         activation='softmax',
         trainable=True):
    
    with tf.name_scope('ENet'):
        enet_out = enet_encoder(inputs, trainable=trainable)
        enet_out = enet_decoder(enet_out, out_channels,
                                activation=activation, trainable=trainable)
        return enet_out
    
    
    



def discriminator(inputs,
                  trainable=True):

    with tf.name_scope('discriminator'):
        conv1 = LeakyReLU()(Conv2D(64, (4,4), strides=2, trainable=trainable)(inputs))
        conv1 = BatchNormalization()(conv1)

        conv2 = LeakyReLU()(Conv2D(128, (4,4), strides=2, trainable=trainable)(conv1))
        conv2 = BatchNormalization()(conv2)

        conv3 = LeakyReLU()(Conv2D(512, (4,4), strides=2, trainable=trainable)(conv2))

        flat_vec = Flatten()(conv3)

        output = Dense(1, activation=None)(flat_vec)

        return output
