from bs4 import BeautifulSoup

def get_blog_contexts(html, is_eval=False):
    contexts = []
    soup = BeautifulSoup(html, 'lxml')
    body = soup.find('body')
    context_table = body.find('table', id='printPost1')
    if context_table is not None:
        context_tag = context_table.find('td', class_='bcc')
        if context_tag is not None:
            contexts = [p_tag.text for p_tag in context_tag.find_all('p')]
    return contexts

class Blog_Converter(object):
    def __init__(self, max_paragraph_length=462):
        self.max_paragraph_length = max_paragraph_length
        self.sep_token = ' | '

        ## 블로그 단락 표시
        self.blog_seperator = '\u200b'

    def convert_html(self, html):
        contexts = get_blog_contexts(html, is_eval=True)
        paragraphs = self.merge_contexts(contexts)
        paragraphs = self.merge_paragraphs_by_len(paragraphs)
        return paragraphs

    def merge_contexts(self, contexts):
        paragraphs = []
        
        temp_paragraph = []
        for context in contexts:
            if self.blog_seperator in context:
                if len(temp_paragraph)>0:
                    paragraphs.append(" ".join(temp_paragraph).strip())
                    temp_paragraph = []
            else:
                if context is not '':
                    temp_paragraph.append(context)

        if len(temp_paragraph) > 0:
            paragraphs.append(" ".join(temp_paragraph).strip())
        return paragraphs

    def merge_paragraphs_by_len(self, paragraphs):
        modified_paragraphs = []

        temp_paragraph = None
        for paragraph in paragraphs:
            if temp_paragraph is None:
                temp_paragraph = paragraph
            else:
                if len(temp_paragraph)<self.max_paragraph_length:
                    temp_paragraph = self.sep_token.join([temp_paragraph, paragraph]).strip()
                else:
                    modified_paragraphs.append(temp_paragraph)
                    temp_paragraph = paragraph
        if temp_paragraph is not None:
            modified_paragraphs.append(temp_paragraph)

        return modified_paragraphs