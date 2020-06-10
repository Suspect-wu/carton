import numpy as np
from PIL import Image
import glob
from scipy.spatial import distance

def process_photo_to_array(photo_loc):
    photo_list = []
    labels = []
    for photo in glob.glob(photo_loc):
        im = Image.open(photo)
        im = np.asarray(im)
        photo_list.append(im)
        if '海绵宝宝' in photo.split('.')[0] :
            labels.append(0)
        else :
            labels.append(1)
    photo_list = np.array(photo_list, dtype='float64')
    photo_list = np.reshape(photo_list, (photo_list.shape[0], 256*256*3))
    labels = np.array(labels, dtype='float64')
    return photo_list, labels

def distance(a, b):
    return distance.euclidean(a.array(ndmin=1),b.array(ndmin=1))

def classify(x_tr, tr_label, x_te, te_label, k=3):
    distance_list = []
    n = len(tr_label)
    for i in range(n):
        distance_list.append(distance(x_tr[i], x_te))
    distance_list = np.array(distance_list)
    arg_dis = np.argsort(distance_list)
    dict = {}
    for i in range(k):
        dict[tr_label[arg_dis[i]]] = dict.get(dict[tr_label[arg_dis[i]]], 0) + 1
    for key, value in dict.items():
        if value == max(dict.values()):
            print(key)

if __name__ == '__main__' :
    train_photo_loc = 'D:\\train\\*.jpg'
    test_photo_loc = 'D:\\test\\*jpg'
    X_tr, tr_label = process_photo_to_array(train_photo_loc)
    X_te, te_label = process_photo_to_array(test_photo_loc)
    print(X_te[0], len(X_te[0]))
    classify(X_tr, tr_label, X_te[0], te_label[0])