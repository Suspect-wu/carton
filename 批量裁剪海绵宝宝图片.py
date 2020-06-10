from PIL import Image
import numpy as np
import os
import glob

def resizePhoto(Unpro_loc, save_loc, heigth=256, width=256):
    UnprocessPhoto = Image.open(Unpro_loc)
    UnprocessPhoto = UnprocessPhoto.convert('RGB')
    NewPhoto =UnprocessPhoto.resize((heigth,width), Image.ANTIALIAS)#Image.ANTALIAS 调整清晰度
    NewPhoto.save(os.path.join(save_loc,os.path.basename(Unpro_loc)))

if __name__ == '__main__':
    if os.path.exists('D:\\process_classify') != 1:
        os.mkdir('D:\\process_classify')
    for Unpro_loc in glob.glob('D:\\kenan\\*.jpg'):
        try:
            resizePhoto(Unpro_loc, 'D:\\process_classify')
        except OSError:
            continue
