import asyncio
from src.crawler import NovoxCrawler

async def main():
    scraper = NovoxCrawler()
    await scraper.start()

if __name__ == "__main__":
    asyncio.run(main())