import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import csv
from sklearn.svm import SVC
from time import  *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
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


def get_pac(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    low_data = pca.transform(data)
    print(len(low_data[1]))
    low_data = scale(low_data)
    print(pca.explained_variance_ratio_)
    return low_data

def get_best_paras(x_train, x_test, y_train, y_test, classname):
    lsvc = SVC()
    params_dict = {'C':np.array([0.1, 1, 10, 15, 20]), 'gamma':np.array([ 0.005, 0.01, 0.05, 0.1, 0.2])}
    skf = StratifiedKFold(n_splits=10)
    GCV = GridSearchCV(lsvc, param_grid=params_dict, cv=skf)
    gcv_result = GCV.fit(x_train, y_train)
    start = time()
    pre_label = gcv_result.predict(x_test)
    print('time used:', time() - start)
    print(gcv_result.best_params_, gcv_result.best_score_, gcv_result.score(x_test,y_test))
    print(classification_report(y_test, pre_label, target_names=classname))
    print(gcv_result.cv_results_)



if __name__ == '__main__':
    photos, labels = read_csv(csv_name)
    photo_list, labels, list = get_photo(photos, labels)
    start = time()
    low_data = get_pac(photo_list, 11)
    print('time used:', time()-start)
    tr_photo = low_data[:1000, :]
    tr_labels = labels[:1000]
    te_photo = low_data[1000:, :]
    te_labels = labels[1000:]
    n = len(te_labels)
    class_name = ['海绵宝宝', '皮卡丘', '香蕉', '小黄人', '柯南']
    get_best_paras(tr_photo, te_photo, tr_labels, te_labels, class_name)

