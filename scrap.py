## ko-wiki 에서 질문에 맞는 데이터를 가져오는 프로그램
import requests
from bs4 import BeautifulSoup

SEARCH_URL = 'https://ko.wikipedia.org/w/index.php'
CONTENT_URL = 'https://ko.wikipedia.org'
PARAMS = {'search':'', 'title':'특수:검색', 'go':'보기', 'ns0':1}

def get_wiki_urls(question:str):
    PARAMS['search'] = question
    response = requests.get(SEARCH_URL, params=PARAMS)
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
    temp_url = CONTENT_URL + url
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