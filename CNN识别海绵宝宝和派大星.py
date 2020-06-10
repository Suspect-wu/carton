import tensorflow as tf
import os
import glob

num_epochs = 20
batch_size = 69
learning_rate = 0.0005
data_dir = 'D:\\tensorflow'
train_h_dir = data_dir + '\\train_h\\*.jpg'
train_p_dir = data_dir + '\\train_p\\*.jpg'
test_h_dir = data_dir + '\\test_h\\*.jpg'
test_p_dir = data_dir + '\\test_p\\*.jpg'

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.cast(image_decoded, dtype=tf.float32)/255.0
    return image_resized, label

if __name__ == '__main__':
    train_h_filenames = tf.constant([filename for filename in glob.glob(train_h_dir)])
    train_p_filenames = tf.constant([filename for filename in glob.glob(train_p_dir)])
    train_filenames = tf.concat([train_h_filenames, train_p_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_h_filenames.shape, dtype=tf.int32),
        tf.ones(train_p_filenames.shape, dtype=tf.int32)],
        axis=-1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=400)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=num_epochs)

    test_h_filenames = tf.constant([filename for filename in glob.glob(test_h_dir)])
    test_p_filenames = tf.constant([filename for filename in glob.glob(test_p_dir)])
    test_filenames = tf.concat([test_h_filenames, test_p_filenames], axis=-1)
    test_labels = tf.concat([
        tf.zeros(test_h_filenames.shape, dtype=tf.int32),
        tf.ones(test_p_filenames.shape, dtype=tf.int32)],
        axis=-1)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(batch_size)

    print(model.metrics_names)
    print(model.evaluate(test_dataset))