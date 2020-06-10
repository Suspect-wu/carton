from lxml import etree
import requests
url = 'https://www.baidu.com/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'}
html = requests.get(url, headers=headers)
selecetor = etree.HTML(html.text)
photo_url = 'https:' + selecetor.xpath('//*[@id="lg"]/img[1]/@src')[0]
photo_b = requests.get(photo_url).content
with open('D:\\baidu.jpg', 'wb') as f:
    f.write(photo_b)
print('0')
##下载文件保存
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
a = requests.get(url)
with open('D:\\haah.csv','wb')as f:
    f.write(a.content)