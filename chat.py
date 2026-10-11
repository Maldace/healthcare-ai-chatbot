import requests
from bs4 import BeautifulSoup
from time import sleep
from random import uniform


# url = 'https://www.vinmec.com/vie/co-the-nguoi/'
# reqs = requests.get(url)
# soup = BeautifulSoup(reqs.text, 'html.parser')

# original_url = 'https://www.vinmec.com'
# urls = []
# ul = soup.find('ul', class_='list_drug flex first-hide')
# print(ul)
# lis = ul.find_all('li', class_=False)
# print(lis)
# for li in lis:
#     align = li.find('div', class_='flex align-item-center')
#     one = align.find('div', class_='flex-one')
#     href = one.find('a', class_='name_drug').get('href')
#     print(href)
    
# url = 'https://www.vinmec.com/vie/benh/ap-xe-nao-3205'
# resp = requests.get(url)
# soup = BeautifulSoup(resp.text, 'html.parser')
# page = soup.find('div', class_='content_detail_sick')
# paragraphs = page.find_all('div', class_='item_detial_sick')
# content = ''
# for txt in paragraphs:
#     pos = txt.text.find('Xem thêm')
#     if pos != -1:
#         content += ("# "+txt.text.strip()[:pos-1]+'\n')
#     else:
#         content += ("# "+txt.text.strip()+'\n')
# with open("text.txt", "a", encoding='utf-8') as f:
#     f.write(content)

count = 0
with open("links.txt", "r") as fp:
    for line in fp:
        url = line.strip()
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        page = soup.find('div', class_='content_detail_sick')
        paragraphs = page.find_all('div', class_='item_detial_sick')
        content = ''
        for txt in paragraphs:
            pos = txt.text.find('Xem thêm')
            if pos != -1:
                content += ("# "+txt.text.strip()[:pos-1]+'\n')
            else:
                content += ("# "+txt.text.strip()+'\n')
        with open("text.txt", "a", encoding='utf-8') as f:
            f.write(content)
        x = uniform(1,5)
        count +=1
        print(f'Đã viết bài thứ {count} trên tổng 678 bài, chờ {x} giây để viết bài tiếp theo')
        sleep(x)