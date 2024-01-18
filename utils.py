import json
import requests
import io

from datetime import datetime
from dateutil.relativedelta import relativedelta

from bs4 import BeautifulSoup
from llama_index import SimpleDirectoryReader
from readability import Document

NEWS_FOLDER = 'docs/news'


def get_latest_news():
    """
    Load documents for a directory in a format that can be used by llama-index
    :return:
    """
    return SimpleDirectoryReader(NEWS_FOLDER).load_data()


def fetch_latest_news(subject="microsoft"):
    """
    Download the latest news from the newsapi.org API
    :param subject:
    :return:
    """
    now = datetime.now()
    one_month_ago = (now - relativedelta(months=1)).strftime("%Y-%m-%d")
    current_date = now.strftime("%Y-%m-%d")

    url = (f'https://newsapi.org/v2/everything?q={subject}&searchIn=title&'
           f'pageSize=10&language=en&from={one_month_ago}&to={current_date}&'
           f'sortBy=popularity&apiKey=bba3bd2dffb2430a903834b58d8b5f90&'
           f'domains=washingtonpost.com,usatoday.com,latimes.com,nypost.com,nytimes.com,news.google.com')
    response = requests.get(url)
    if response.status_code != 200:
        print(response.content)
        raise Exception("Could not get latest news.")
    else:
        articles = [article for article in json.loads(response.text)["articles"]]
        return get_full_content(articles)


def get_full_content(articles):
    """
    Get the full content of the articles, since it is truncated in the newsapi.org API response
    :param articles:
    :return:
    """
    plain_articles = []
    for article in articles:
        page = article['url']
        res = requests.get(page)
        doc = Document(res.content)
        print(article['url'])
        print(article['content'])
        # Parse the content of the request with BeautifulSoup
        soup = BeautifulSoup(doc.summary(), 'html.parser')
        # Extract the main content of the article
        main_content = soup.find('body').get_text()
        article['content'] = main_content
        # Convert the string to a plain text string
        plain_articles.append('\n'.join(f'{k}: {v}' for k, v in article.items()))
    return plain_articles


def download_latest_news(subject="microsoft"):
    """
    Downloads the latest news on a subject.
    :param subject: the subject of the news to download
    :return:
    """
    for i, value in enumerate(fetch_latest_news(subject)):
        with io.open(NEWS_FOLDER + f"/news_{subject}_{i + 1}.txt", "w", encoding="utf-8") as f:
            f.write(value)

    return "News downloaded successfully."

# download_latest_news("google")
