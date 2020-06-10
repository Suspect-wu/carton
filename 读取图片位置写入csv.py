import random
import csv
import os
import glob


def load_csv(root, csv_name):
    if not os.path.exists(os.path.join('D:', csv_name)):
        photo_list = []
        photo_list += glob.glob(os.path.join(root, '*.jpg'))
        random.shuffle(photo_list)
        labels = []
        for photo in photo_list:
            slice_photo = photo.split('.')[0]
            if '海绵宝宝' in slice_photo:
                labels.append(0)
            elif '皮卡丘' in slice_photo:
                labels.append(1)
            elif '香蕉' in slice_photo:
                labels.append(2)
            elif '小黄人' in slice_photo:
                labels.append(3)
            else:
                labels.append(4)
        with open(os.path.join('D:', csv_name), 'w', encoding= 'utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(photo_list)):
                writer.writerow([photo_list[i], labels[i]])
        print('csv done :', os.path.join('D:', csv_name))

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

if __name__ == '__main__':
    root = 'D:\\process_classify'
    csv_name = 'location.csv'
    load_csv(root, csv_name)
    read_csv(csv_name)


