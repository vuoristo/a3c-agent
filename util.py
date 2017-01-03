import tensorflow as tf

def _kernel_img_summary(img, shape, name):
  xx = shape[0] + 2
  yy = shape[1] + 2

  tmp = tf.slice(img,(0,0,0,0),(-1,-1,1,-1))
  tmp = tf.reshape(tmp,(shape[0],shape[1],shape[3]))
  tmp = tf.image.resize_image_with_crop_or_pad(tmp,xx,yy)
  tmp = tf.reshape(tmp,(xx,yy,4,4))
  tmp = tf.transpose(tmp,(2,0,3,1))
  tmp = tf.reshape(tmp,(1,4*xx,4*yy,1))

  return tf.image_summary(name, tmp)

def _activation_summary(img, shape, name):
  xx = shape[0] + 1
  yy = shape[1] + 1

  tmp = tf.slice(img, (0,0,0,0), (1,-1,-1,-1))
  tmp = tf.reshape(tmp, shape)
  tmp = tf.image.resize_image_with_crop_or_pad(tmp, xx, yy)
  tmp = tf.reshape(tmp,(xx,yy,4,4))
  tmp = tf.transpose(tmp, (2,0,3,1))
  tmp = tf.reshape(tmp, (1,4*xx,4*yy,1))

  return tf.image_summary('activation image', tmp)
