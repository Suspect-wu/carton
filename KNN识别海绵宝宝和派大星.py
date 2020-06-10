import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import classification_report
from PIL import Image
plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['axes.unicode_minus']=False
myfont = fm.FontProperties(fname=r'D:\Fonts\simkai.ttf')
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
    photo_list = np.array(photo_list, dtype='int32')
    labels_list = np.array(labels_list, dtype='int32')
    pro_photo_list = np.array(photo_list, dtype='float64')
    pro_photo_list = np.reshape(pro_photo_list, (pro_photo_list.shape[0], 256*256*3))
    labels_list = np.array(labels_list, dtype=np.int)
    return pro_photo_list, labels_list, photo_list

if __name__ == '__main__' :
    photos, labels = read_csv(csv_name)
    photo_list, labels, list = get_photo(photos, labels)
    photo_list = scale(photo_list)
    tr_photo = photo_list[:1000, :]
    tr_labels = labels[:1000]
    te_photo = photo_list[1000:, :]
    te_labels = labels[1000:]
    n = len(te_labels)
    class_name = ['海绵宝宝', '皮卡丘', '香蕉', '小黄人', '柯南']
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, 1+i)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_name[int(labels[i])],fontsize=12)
        plt.imshow(list[i], cmap=plt.cm.binary)
        plt.grid(False)
    plt.show()
    for i in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn.fit(tr_photo, tr_labels)
        predict_labels = knn.predict(te_photo)
        print('k={}时'.format(i), classification_report(te_labels, predict_labels, target_names=class_name))





