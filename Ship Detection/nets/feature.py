from __future__ import print_function

from keras import layers
from keras.initializers import random_normal, constant
from keras.layers import (Activation, BatchNormalization, Conv2D, Dropout,Reshape,AveragePooling2D,MaxPooling2D,Concatenate,Lambda)
from keras.regularizers import l2
from deform_conv.layers import ConvOffset2D

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), kernel_initializer=random_normal(stddev=0.02), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size,padding='same', kernel_initializer=random_normal(stddev=0.02), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), kernel_initializer=random_normal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer=random_normal(stddev=0.02),
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer=random_normal(stddev=0.02),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), kernel_initializer=random_normal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer=random_normal(stddev=0.02),
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def HFF(inputs):

    x = Conv2D(16, (7, 7), strides=(2, 2), kernel_initializer=random_normal(stddev=0.02), padding="same",name='conv1', use_bias=False)(inputs)
    x_1 = BatchNormalization(name='bn_conv1')(x)

    x_2 = conv_block(x_1, 3, [16, 16, 32], stage=2, block='a')
    x_2 = identity_block(x_2, 3,[16, 16, 32],stage=2, block='b')
    x_2 = identity_block(x_2, 3, [16, 16, 32],stage=2, block='c')

    x_3 = conv_block(x_2, 3, [32, 32, 64], stage=3, block='a')
    x_3 = identity_block(x_3, 3, [32, 32, 64], stage=3, block='b')
    x_3 = identity_block(x_3, 3, [32, 32, 64], stage=3, block='c')
    x_3 = identity_block(x_3, 3, [32, 32, 64], stage=3, block='d')

    x_4 = conv_block(x_3, 3, [64, 64, 128], stage=4, block='a')
    x_4 = identity_block(x_4, 3, [64, 64, 128], stage=4, block='b')
    x_4 = identity_block(x_4, 3, [64, 64, 128], stage=4, block='c')
    x_4 = identity_block(x_4, 3, [64, 64, 128], stage=4, block='d')
    x_4 = identity_block(x_4, 3, [64, 64, 128], stage=4, block='e')
    x_4 = identity_block(x_4, 3, [64, 64, 128], stage=4, block='f')

    x_5 = conv_block(x_4, 3, [128, 128, 256], stage=5, block='a')
    x_5 = identity_block(x_5, 3, [128, 128, 256], stage=5, block='b')
    x_5 = identity_block(x_5, 3, [128, 128, 256], stage=5, block='c')

    x_3_P1 = layers.Conv2DTranspose(32, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x_3)
    x_2_P1 = Conv2D(32, (3, 3), strides=(1, 1),padding="same")(x_2)
    P1 = layers.add([x_2_P1, x_3_P1])
    x_4_P2 = layers.Conv2DTranspose(64, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x_4)
    x_3_P2 = Conv2D(64, (3, 3), strides=(1, 1),padding="same")(x_3)
    P2 = layers.add([x_3_P2, x_4_P2])
    x_5_P3 = layers.Conv2DTranspose(128, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x_5)
    x_4_P3 =Conv2D(128, (3, 3), strides=(1, 1),padding="same")(x_4)
    P3=layers.add([x_4_P3, x_5_P3])
    x_P2_L1 =layers.Conv2DTranspose(64, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(P2)
    x_P1_L1 =Conv2D(64, (3, 3), strides=(1, 1),padding="same")(P1)
    x_2_L1 =Conv2D(64, (3, 3), strides=(1, 1),padding="same")(x_2)
    L1=layers.add([x_P1_L1, x_P2_L1,x_2_L1])
    x_P3_L2 =layers.Conv2DTranspose(128, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(P3)
    x_P2_L2 =Conv2D(128, (3, 3), strides=(1, 1),padding="same")(P2)
    L2=layers.add([x_P2_L2, x_P3_L2])
    x_L2_I =layers.Conv2DTranspose(128, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(L2)
    x_L1_I =Conv2D(128, (3, 3), strides=(1, 1),padding="same")(L1)
    x_2_I =Conv2D(128, (3, 3), strides=(1, 1),padding="same")(x_2)
    I=layers.add([x_L1_I, x_L2_I,x_2_I])
    print("I.shape")
    print(I.shape)

    return I


def DFR_head(x,num_classes):
    x = Dropout(rate=0.2)(x)
    x_1 = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02))(x)
    x_1 = Conv2D(128, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02))(x_1)

    y_1 = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02))(x_1)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)
    print('y_1.shape')
    print(y_1.shape)
    y1_D = ConvOffset2D(32)(y_1)
    print('y1_D.shape')
    print(y1_D.shape)
    y1 = Conv2D(num_classes, 1, kernel_initializer=constant(0), bias_initializer=constant(-2.19), activation='sigmoid')(y1_D)
    print('y1.shape')
    print(y1.shape)
    # wh header
    y_2 = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02))(x_1)
    y_2 = BatchNormalization()(y_2)
    y_2 = Activation('relu')(y_2)
    y2_fuse = layers.add([y1_D,y_2])
    y2 = Conv2D(2, 1, kernel_initializer=random_normal(stddev=0.02))(y2_fuse)

    # reg header
    y_3 = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer=random_normal(stddev=0.02))(x_1)
    y_3 = BatchNormalization()(y_3)
    y_3 = Activation('relu')(y_3)
    y3 = Conv2D(2, 1, kernel_initializer=random_normal(stddev=0.02))(y_3)
    return y1, y2, y3
