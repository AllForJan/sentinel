# coding: utf-8
import tensorflow as tf

concat = tf.keras.layers.concatenate
K = tf.keras.backend
binary_crossentropy = tf.keras.backend.binary_crossentropy
SMOOTH = 1e-12


class UpSampling2D(tf.keras.layers.UpSampling2D):
  """Implements bilinear upsampling, since the original uses only nearest neighbor.
  
  Arguments:
    tf {[type]} -- [description]
  
  Returns:
    [type] -- [description]
  """

  def __init__(self, scale_factor, mode='nearest'):
    super(UpSampling2D, self).__init__(size=scale_factor)
    self._mode = mode

  def call(self, x):
    original_shape = K.int_shape(x)
    new_shape = tf.shape(x)[1:3]
    height_factor, width_factor = self.size[0], self.size[1]
    new_shape *= tf.to_int32([height_factor, width_factor])
    # x = tf.image.resize_nearest_neighbor(x, new_shape)  # original
    method = tf.image.ResizeMethod.BILINEAR if self._mode == 'bilinear' else \
        tf.image.ResizeMethod.NEAREST_NEIGHBOR
    x = tf.image.resize_images(images=x, size=new_shape, method=method)
    x.set_shape((None, original_shape[1] * height_factor
                 if original_shape[1] is not None else None,
                 original_shape[2] * width_factor
                 if original_shape[2] is not None else None, None))
    return x


def ConvRelu(out):
  return tf.keras.Sequential([
      tf.keras.layers.ZeroPadding2D(padding=1),
      tf.keras.layers.Conv2D(filters=out, kernel_size=3)
  ])


def DecoderBlockV2(middle_channels, out_channels, is_deconv):
  if is_deconv:
    """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
        """

    return tf.keras.Sequential([
        ConvRelu(middle_channels),
        tf.keras.layers.ConvTranspose2d(
            out_channels, kernel_size=4, stride=2, padding=1),
        tf.keras.layers.Activation(activation='relu')
    ])
  else:
    return tf.keras.Sequential([
        UpSampling2D(scale_factor=2, mode='bilinear'),
        ConvRelu(middle_channels),
        ConvRelu(out_channels)
    ])


def vgg16_encoder(input_shape, freeze=False):
  vgg16 = tf.keras.applications.VGG16(
      weights='imagenet', include_top=False, input_shape=input_shape)
  if freeze:
    for layer in vgg16.layers:
      layer.trainable = False
  return vgg16


def bn_elu(layers, use=True):
  if use:
    layers.insert(1, BatchNormalization(mode=0, axis=1))
    layers.insert(2, keras.layers.advanced_activations.ELU())
    layers.insert(4, BatchNormalization(mode=0, axis=1))
    layers.insert(5, keras.layers.advanced_activations.ELU())
  return layers


def vgg16_decoder(vgg16, num_filters, num_classes, is_deconv, use_bn_elu):
  x = vgg16.input

  conv1 = tf.keras.Sequential(bn_elu(vgg16.layers[1:3], use=use_bn_elu))(x)
  x = vgg16.layers[3](conv1)  # pool

  conv2 = tf.keras.Sequential(bn_elu(vgg16.layers[4:6], use=use_bn_elu))(x)
  x = vgg16.layers[6](conv2)  # pool

  conv3 = tf.keras.Sequential(bn_elu(vgg16.layers[7:10], use=use_bn_elu))(x)
  x = vgg16.layers[10](conv3)  # pool

  conv4 = tf.keras.Sequential(bn_elu(vgg16.layers[11:14], use=use_bn_elu))(x)
  x = vgg16.layers[14](conv4)  # pool

  conv5 = tf.keras.Sequential(bn_elu(vgg16.layers[15:18], use=use_bn_elu))(x)
  x = vgg16.layers[18](conv5)  # pool

  center = DecoderBlockV2(num_filters * 8 * 2, num_filters * 8, is_deconv)
  x = center(x)

  dec5 = DecoderBlockV2(num_filters * 8 * 2, num_filters * 8, is_deconv)
  x = dec5(concat([x, conv5]))

  dec4 = DecoderBlockV2(num_filters * 8 * 2, num_filters * 8, is_deconv)
  x = dec4(concat([x, conv4]))

  dec3 = DecoderBlockV2(num_filters * 4 * 2, num_filters * 2, is_deconv)
  x = dec3(concat([x, conv3]))

  dec2 = DecoderBlockV2(num_filters * 2 * 2, num_filters, is_deconv)
  x = dec2(concat([x, conv2]))

  dec1 = ConvRelu(num_filters)
  x = dec1(concat([x, conv1]))

  final = tf.keras.layers.Conv2D(num_classes, kernel_size=1, name='logit')

  return final(x)


def vgg16_unet(hparams):
  input_shape = (hparams.image_size, hparams.image_size, 3)
  vgg16 = vgg16_encoder(input_shape, hparams.freeze_encoder)
  logit = vgg16_decoder(vgg16, hparams.num_classes, hparams.num_filters,
                        hparams.is_deconv, hparams.use_bn_elu)
  ternaus_net = tf.keras.Model(vgg16.input, logit)
  return ternaus_net


def jaccard_coef(y_true, y_pred):
  intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
  sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

  jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)

  return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
  y_pred_pos = K.round(K.clip(y_pred, 0, 1))

  intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
  sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

  jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)

  return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
  return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(
      y_pred, y_true)


# def get_unet0():
#   inputs = Input((num_channels, img_rows, img_cols))
#   conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
#   conv1 = BatchNormalization(mode=0, axis=1)(conv1)
#   conv1 = keras.layers.advanced_activations.ELU()(conv1)
#   conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
#   conv1 = BatchNormalization(mode=0, axis=1)(conv1)
#   conv1 = keras.layers.advanced_activations.ELU()(conv1)
#   pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#   conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
#   conv2 = BatchNormalization(mode=0, axis=1)(conv2)
#   conv2 = keras.layers.advanced_activations.ELU()(conv2)
#   conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
#   conv2 = BatchNormalization(mode=0, axis=1)(conv2)
#   conv2 = keras.layers.advanced_activations.ELU()(conv2)
#   pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#   conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
#   conv3 = BatchNormalization(mode=0, axis=1)(conv3)
#   conv3 = keras.layers.advanced_activations.ELU()(conv3)
#   conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
#   conv3 = BatchNormalization(mode=0, axis=1)(conv3)
#   conv3 = keras.layers.advanced_activations.ELU()(conv3)
#   pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#   conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
#   conv4 = BatchNormalization(mode=0, axis=1)(conv4)
#   conv4 = keras.layers.advanced_activations.ELU()(conv4)
#   conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
#   conv4 = BatchNormalization(mode=0, axis=1)(conv4)
#   conv4 = keras.layers.advanced_activations.ELU()(conv4)
#   pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#   conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
#   conv5 = BatchNormalization(mode=0, axis=1)(conv5)
#   conv5 = keras.layers.advanced_activations.ELU()(conv5)
#   conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
#   conv5 = BatchNormalization(mode=0, axis=1)(conv5)
#   conv5 = keras.layers.advanced_activations.ELU()(conv5)

#   up6 = merge(
#       [tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4],
#       mode='concat',
#       concat_axis=1)
#   conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
#   conv6 = BatchNormalization(mode=0, axis=1)(conv6)
#   conv6 = keras.layers.advanced_activations.ELU()(conv6)
#   conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
#   conv6 = BatchNormalization(mode=0, axis=1)(conv6)
#   conv6 = keras.layers.advanced_activations.ELU()(conv6)

#   up7 = merge(
#       [tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3],
#       mode='concat',
#       concat_axis=1)
#   conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
#   conv7 = BatchNormalization(mode=0, axis=1)(conv7)
#   conv7 = keras.layers.advanced_activations.ELU()(conv7)
#   conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
#   conv7 = BatchNormalization(mode=0, axis=1)(conv7)
#   conv7 = keras.layers.advanced_activations.ELU()(conv7)

#   up8 = merge(
#       [tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2],
#       mode='concat',
#       concat_axis=1)
#   conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
#   conv8 = BatchNormalization(mode=0, axis=1)(conv8)
#   conv8 = keras.layers.advanced_activations.ELU()(conv8)
#   conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
#   conv8 = BatchNormalization(mode=0, axis=1)(conv8)
#   conv8 = keras.layers.advanced_activations.ELU()(conv8)

#   up9 = merge(
#       [tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1],
#       mode='concat',
#       concat_axis=1)
#   conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
#   conv9 = BatchNormalization(mode=0, axis=1)(conv9)
#   conv9 = keras.layers.advanced_activations.ELU()(conv9)
#   conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
#   crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
#   conv9 = BatchNormalization(mode=0, axis=1)(crop9)
#   conv9 = keras.layers.advanced_activations.ELU()(conv9)
#   conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid')(conv9)

#   model = Model(input=inputs, output=conv10)

#   return model
