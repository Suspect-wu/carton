import requests
import json
import os


def GetmangPage(key, page):
    params = []
    for i in range(30, 30+30*page, 30):
        params.append({
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': key,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': '',
            'z': 0,
            'ic': '',
            'hd':'',
            'latest':'',
            'copyright':'',
            'word': key,
            's': '',
            'se': '',
            'tab': '',
            'width': 0,
            'height': 0,
            'face': '',
            'istype': '',
            'qc': '',
            'nc': '',
            'fr': '',
            'expermode':'',
            'force':'',
            'pn': i,
            'rn': 30,
            'gsm': '',
            '1575128764215': ''
        })
    url = 'https://image.baidu.com/search/acjson'
    urls = []
    for i in params:
        try:
            urls.append(requests.get(url, params=i).json(strict=False).get('data'))
        except json.JSONDecodeError:
            continue
    print(len(urls))
    return urls

def getphotos(path, urls):
    if os.path.exists(path) == 0:
        os.mkdir(path)
    numphoto = 500
    for url in urls:
        for i in url:
            if i.get('thumbURL') == None:
                print('该链接照片不存在')
            else:
                print('正在下载第'+'{}'.format(numphoto)+'张照片')
                photo_url = i.get('thumbURL')
                new_path = path + '\\' +'柯南'+str(numphoto) +'.jpg'
                b_photos = requests.get(photo_url)
                with open(new_path, 'wb') as f:
                    f.write(b_photos.content)
                numphoto += 1


if __name__ == '__main__' :
    name = '柯南照片头像'
    page = 3
    a = GetmangPage(name, page)
    path = 'D:\\classify'
    b = getphotos(path, a)