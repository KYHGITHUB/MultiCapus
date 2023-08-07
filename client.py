import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
import pandas as pd
pd.set_option('display.max_columns', None)      # 데이터프레임 끝까지 보여주기
pd.set_option('display.max_rows', None)


'''
with open('test.html', encoding = 'utf=8') as f:
    line = f.read()
line
soup = bs(line, 'html.parser')
print(soup)
print(soup.prettify())
print(soup.children)
soup_children_list = list(soup.children)
list(soup.children)[1]
print(soup_children_list)
print(soup.body)
print(soup.head)
print(soup.find_all('p'))
for i in soup.find_all('p'):
    print(i.text)

news = ' https://news.daum.net'
soup = bs(urlopen(news), 'html.parser')
print(soup)
print(soup.find_all('div', {'class':'item_issue'}))    # div 태그에서 class(key)값이 item_issue(value)인것을 찾아라
for i in soup.find_all('div', {'class':'item_issue'}):
    print(i.text)
print(soup.find_all('a')[:5])
for i in soup.find_all('a'):
    print(i.get('href'))
for i in soup.find_all('div', {'class':'item_issue'}):
    print(i.find_all('a')[0].get('href'))
    
article1 = 'https://go.seoul.co.kr/news/newsView.php?id=20200427004004&wlog_tag3=daum'
soup = bs(urlopen(article1).read(), 'html.parser')
print(soup)
for i in soup.find_all('p'):
    print(i.text)
    
news = ' https://news.daum.net'
soup = bs(urlopen(news), 'html.parser')
headline = soup.find_all('div', {'class':'item_issue'})
for i in headline:
    print(i)
for i in headline:
    print(i.text)
    soup3 = bs(urlopen(i.find_all('a')[0].get('href')).read(), 'html.parser')
    for j in soup3.find_all('p'):
        print(j.text)
for i in soup.find_all('div', {'class':'item_issue'}):
    print(i.find_all('a')[0].get('href'))
with open('link.txt', 'w') as f:
    for i in soup.find_all('div', {'class':'item_issue'}):
        f.write(i.find_all('a')[0].get('href')+'\n')

article1 = 'https://v.daum.net/v/20230807102700905'
soup = bs(urlopen(article1).read(), 'html.parser')
with open('article1.txt', 'w') as f:
    for i in soup.find_all('p'):        # p태그는 기사내용을 찾기위함이다
        f.write(i.text+'\n')

url = 'https://news.daum.net/'
soup = bs(urlopen(url).read(), 'html.parser')
headline = soup.find_all('div', {'class':'item_issue'})
headline

for i in headline:
    print(i.text)
for i in headline:
    print(i.text)
    print(i.find_all('a')[0].get('href'))
    new_url = i.find_all('a')[0].get('href')
    soup2 = bs(urlopen(new_url).read(), 'html.parser')
    for j in soup2.find_all('p'):
        print(j.text+'\n')

with open('article_total.txt', 'w',encoding = 'utf-8') as f:
    for i in headline:
        f.write(i.text)
        f.write(i.find_all('a')[0].get('href')+'\n')
        new_url = i.find_all('a')[0].get('href')
        soup2 = bs(urlopen(new_url).read(), 'html.parser')
        for j in soup2.find_all('p'):
            f.write(j.text+'\n')
'''
url = 'https://www.chicagomag.com/chicago-magazine/january-2023/our-30-favorite-things-to-eat-right-now/'
hdr = {'User-Agent':'Mozilla/5.0'}
req = Request(url, headers=hdr)
page = urlopen(req)
soup = bs(page, 'html.parser')
soup
temp = soup.find_all('div', {'class':'article-body'})[0]
temp
f_list = []
r_list = []
p_list = []
a_list = []
temp.find_all('h2')[0].text
for f in temp.find_all('h2'):
    f_list.append(f.text)
f_list
temp.find_all('h3')[22].text.split('at')[1]
for r in temp.find_all('h3'):
    r_list.append(r.text.split('at')[1].strip())
r_list
temp.find_all('p')[0].text.index('$')
temp.find_all('p')[0].text[314]
temp.find_all('p')[0].text[314:].split()[0].strip('.')
' '.join(temp.find_all('p')[0].text[314:].split()[1:]).strip()
for p in temp.find_all('p'):
    p_index = p.text.index('$')
    p_list.append(p.text[p_index:].split()[0].strip('.'))
    a_list.append(' '.join(p.text[p_index:].split()[1:]).strip())
p_list
a_list
data = {'Food':f_list, 'Restaurant':r_list, 'Price':p_list, 'Address':a_list}
data
df = pd.DataFrame(data)
print(df)

