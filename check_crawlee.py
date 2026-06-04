import inspect
from crawlee.crawlers import BeautifulSoupCrawler
print(inspect.signature(BeautifulSoupCrawler.__init__))
