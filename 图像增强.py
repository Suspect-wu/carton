import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random

def enhance_photo(photo_loc):
    photo_loc = photo_loc
    photo = tf.io.read_file(photo_loc)
    photo = tf.image.decode_jpeg(photo, channels=3)
    photo = tf.image.resize(photo, [244,244])
    photo = tf.image.random_flip_up_down(photo)
    photo = tf.image.rot90(photo)
    photo = np.uint8(photo)
    photo = Image.fromarray(photo)
    photo.save(os.path.join())
