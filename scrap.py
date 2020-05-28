## ko-wiki, naver-blog 에서 질문에 맞는 데이터를 가져오는 프로그램
import requests
from bs4 import BeautifulSoup


### WIKI 설정
WIKI_SEARCH_URL = 'https://ko.wikipedia.org/w/index.php'
WIKI_CONTENT_URL = 'https://ko.wikipedia.org'
WIKI_PARAMS = {'search':'', 'title':'특수:검색', 'go':'보기', 'ns0':1}

def get_wiki_urls(question:str):
    WIKI_PARAMS['search'] = question
    response = requests.get(WIKI_SEARCH_URL, params=WIKI_PARAMS)
    ## 200이 정상코드입니다.

    searched_urls = []
    if response.status_code == 200:
        body = BeautifulSoup(response.text, 'lxml')
        ul = body.find('ul', class_='mw-search-results')
        a_tags = ul.find_all('div', class_='mw-search-result-heading')

        for a_tag in a_tags:
            if a_tag.a is not None:
                #print(a_tag.a['href'])
                searched_urls.append(a_tag.a['href'])

    return searched_urls

def get_wiki_content(url:str):
    temp_url = WIKI_CONTENT_URL + url
    response = requests.get(temp_url)
    if response.status_code == 200:
        return response.text

    return ''

def get_wiki_data(question:str, num=5):
    assert len(question) > 0 and len(question) < 100, "질문은 1글자 이상, 100글자 이하이어야 합니다."
    urls = get_wiki_urls(question)

    contents = []
    for idx, url in enumerate(urls):
        if idx >=num:
            break
        content = get_wiki_content(url)
        if content is not None and len(content) > 0:
            contents.append(content)
    return contents

### Blog 설정
BLOG_SEARCH_URL = 'https://search.naver.com/search.naver'
BLOG_PARAMS = {'query':'', 'where':'post', 'sm':'tab_jum'}

def get_blog_urls(question:str):
    BLOG_PARAMS['query'] = question
    response = requests.get(BLOG_SEARCH_URL, params=BLOG_PARAMS)
    ## 200이 정상코드입니다.

    searched_urls = []
    if response.status_code == 200:
        body = BeautifulSoup(response.text, 'lxml')
        ul = body.find('ul', id='elThumbnailResultArea')
        a_tags = ul.find_all('li', class_='sh_blog_top')

        for a_tag in a_tags:
            temp_a = a_tag.find('a', class_='_sp_each_url')
            if temp_a is not None:
                searched_urls.append(temp_a['href'])

    return searched_urls

def get_blog_data(question:str, num=5):
    assert len(question) > 0 and len(question) < 100, "질문은 1글자 이상, 100글자 이하이어야 합니다."
    urls = get_blog_urls(question)

    contents = []
    for idx, url in enumerate(urls):
        if idx >=num:
            break
        content = get_blog_content(url)
        if content is not None and len(content) > 0:
            contents.append(content)
    return contents

def get_blog_content(url:str):
    temp_url = url

    ## 네이버 블로그는 iframe에서 source url을 추출하여 사용한다.
    if 'blog.naver.com' in url or 'blog.me' in url:
        temp_url = find_iframe_url(url)

    response = requests.get(temp_url)
    if response.status_code == 200:
        return response.text

    return ''

def find_iframe_url(url):
    iframe_url = url

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        iframe_located = soup.find('iframe')['src']
        if iframe_located is not None and iframe_located is not '':
            iframe_url = 'https://blog.naver.com/' + iframe_located

    return iframe_url