import tensorflow as tf
import glob
import random


learning_rate = 0.0001
num_epochs = 20
batch_size = 15
def getloc(photo_loc):
    photo_list = []
    label = []
    for i in glob.glob(photo_loc):
        photo_list.append(i)
    random.shuffle(photo_list)
    random.shuffle(photo_list)
    return photo_list

def _decode_and_resize(filename, label):
    label = tf.cast(label, dtype=tf.int64)
    im_string = tf.io.read_file(filename)
    im_decode = tf.image.decode_jpeg(im_string)
    im_resized = tf.cast(im_decode, dtype=tf.float32)/255.0
    return  im_resized, label

if __name__ == '__main__':
    filenames_list = getloc(photo_loc = 'D:\\process_classify\\*.jpg')
    labels = []
    for photo in filenames_list:
        slice_photo = photo.split('.')[0]
        if '海绵宝宝' in slice_photo:
            labels.append(0)
        elif '皮卡丘' in slice_photo:
            labels.append(1)
        elif '香蕉' in slice_photo:
            labels.append(2)
        elif '小黄人' in slice_photo:
            labels.append(3)
        else :
            labels.append(4)
    tr_filenames_list = filenames_list[:1200]
    te_filenames_list = filenames_list[1200:]
    tr_labels_list = labels[:1200]
    te_labels_list = labels[1200:]
    te_filenames = tf.constant(te_filenames_list)
    tr_filenames = tf.constant(tr_filenames_list)
    tr_labels = tf.constant(tr_labels_list)
    te_labels = tf.constant(te_labels_list)
    train_dataset = tf.data.Dataset.from_tensor_slices((tr_filenames, tr_labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size=400)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regulizers.l2(0.0001)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=num_epochs)


    test_dataset = tf.data.Dataset.from_tensor_slices((te_filenames, te_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(batch_size)

    print(model.metrics_names)
    print(model.evaluate(test_dataset))
