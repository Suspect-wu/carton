import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import classification_report
import csv
import os
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False
myfont = fm.FontProperties(fname=r'D:\Fonts\simkai.ttf')
learning_rate = 0.001
epochs = 1
class_name = ['海绵宝宝', '皮卡丘', '香蕉', '小黄人', '柯南']
csv_name = 'location.csv'
def read_csv(csv_name):
    photos = []
    labels = []
    with open(os.path.join('D:', csv_name), encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            photo_loc, label= row
            photos.append(photo_loc)
            labels.append(int(label))
    print(labels)
    print(photos)
    return photos, labels

def get_photo(photos, labels):
    photo_filenames = photos
    photo_list = []
    labels_list =labels
    for photo in photo_filenames:
        im = Image.open(photo)
        im = np.array(im)
        photo_list.append(im)
    photo_list = np.array(photo_list, dtype='float32')
    labels_list = np.array(labels_list, dtype='int32')
    print(photo_list.shape, labels_list.shape)
    return photo_list, labels_list

def drew_val_train_accuracy(history, epoch):
    epoch2 = [i for i in range(1, 1+epoch, 1)]
    plt.plot(epoch2, history.history['sparse_categorical_accuracy'], 'b', label='训练集')
    plt.plot(epoch2, history.history['val_sparse_categorical_accuracy'], 'r', label='测试集')
    plt.xticks([i for i in range(1, 1+epoch, 5)])
    plt.xlabel('训练时期')
    plt.ylabel('准确率')
    plt.legend(loc='lower right')
    plt.title('训练集与验证集准确率变化图')
    plt.show()

def drew_val_train_loss(history, epoch):
    epoch2 = [i for i in range(1, 1+epoch, 1)]
    plt.plot(epoch2, history.history['loss'], 'b', label='训练集')
    plt.plot(epoch2, history.history['val_loss'], 'r', label='测试集')
    plt.xticks([i for i in range(1, 1 + epoch, 5)])
    plt.xlabel('训练时期')
    plt.ylabel('损失值')
    plt.legend(loc='lower right')
    plt.title('训练集与验证集损失值变化图')
    plt.show()


def draw_plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions_array)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    if predicted_label == true_label :
        color = 'blue'
    else :
        color = 'red'
    plt.xlabel('T:{}({:2.0f}%{})'.format(class_name[true_label], 100*np.max(predictions_array), class_name[predicted_label]), color=color, fontproperties=myfont)

def draw_value_array(i, predictions_array, true_label):
    predictions_array, true_label= predictions_array[i], true_label[i]
    plt.grid(False)
    plt.bar(class_name, predictions_array)
    plt.ylabel('Prob')




if __name__ == '__main__':
    photos, labels = read_csv(csv_name)
    photo_list, labels_list = get_photo(photos, labels)
    un_photo_list = np.array(photo_list, dtype=np.int)
    photo_list = photo_list/255.0
    tr_photo_list = photo_list[:1000]
    tr_labels_list = labels_list[:1000]
    te_photo_list = photo_list[1000:]
    te_labels_list = labels_list[1000:]
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.summary()
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.001, patience=3)#连续三次验证集测试结果没有提升0.001，就提前结束。
    history = model.fit(tr_photo_list, tr_labels_list, batch_size=5, epochs=epochs, validation_split=0.1)#callbacks=[early_stopping]
    print(history.history)#字典
    model.save('my_model.h5')
    drew_val_train_accuracy(history, epochs)
    drew_val_train_loss(history, epochs)
    result = model.evaluate(te_photo_list, te_labels_list, batch_size=5)
    print(result)
    pedicetions =model.predict(te_photo_list, batch_size=5)
    num_rows = 4
    num_cols = 2
    num_images = num_rows*num_cols
    plt.figure(figsize=(4*4*num_cols, 4*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols*2, i*2+1)
        draw_plot_image(i, pedicetions, te_labels_list, un_photo_list[1000:])
        plt.subplot(num_rows, num_cols*2, 2*i+2)
        draw_value_array(i, pedicetions, te_labels_list)
    plt.show()
    plt.figure(figsize=(4 *2 , 4))
    plt.subplot(1, 2, 1)
    draw_plot_image(-1, pedicetions, te_labels_list, un_photo_list[1000:])
    plt.subplot(1, 2, 2)
    draw_value_array(-1, pedicetions, te_labels_list)
    plt.show()
    plt.figure(figsize=(4 *2 , 4))
    plt.subplot(1, 2, 1)
    draw_plot_image(-2, pedicetions, te_labels_list, un_photo_list[1000:])
    plt.subplot(1, 2, 2)
    draw_value_array(-2, pedicetions, te_labels_list)
    plt.show()
    print(pedicetions)
    pred = []
    for i in range(len(pedicetions)):
        pred.append(np.argmax(pedicetions[i]))
    pred = np.array(pred)
    print(pred)
    print(classification_report(te_labels_list, pred, target_names=class_name))