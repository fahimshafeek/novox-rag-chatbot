import json
import asyncio
import random
from crawlee import ConcurrencySettings
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from src.config import START_URL, MAX_REQUESTS
from src.extractor import extract_and_tag

class NovoxCrawler:
    def __init__(self):
        self.crawler = PlaywrightCrawler(
            max_requests_per_crawl=MAX_REQUESTS,
            headless=True,
            browser_type='chromium',
            concurrency_settings=ConcurrencySettings(
                min_concurrency=1,
                desired_concurrency=1,
                max_concurrency=1
            )
        )
        self.output_file = "scraped_data_output.jsonl"
        with open(self.output_file, "w", encoding="utf-8") as f:
            pass

    def setup_routes(self):
        @self.crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            url = context.request.url
            
            await context.enqueue_links()
            
            # Wait for any potential JS challenges to complete
            await context.page.wait_for_load_state('networkidle')
            html_content = await context.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            processed_page = extract_and_tag(url, soup)
            
            if processed_page:
                print(f"✅ [{processed_page['role'].upper()}] -> {url}")
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(processed_page) + "\n")
            
            # Keep the randomized human jitter to avoid being blocked.
            sleep_time = random.uniform(2.0, 5.0) # Slightly reduced as HTTP is faster than browser
            await asyncio.sleep(sleep_time)

    async def start(self):
        self.setup_routes()
        print(f"🚀 Starting Playwright crawl at: {START_URL}")
        await self.crawler.run([START_URL])
        print(f"\n🎉 Done! Check '{self.output_file}'.")