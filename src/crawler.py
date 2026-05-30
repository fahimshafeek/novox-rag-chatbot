import json
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from src.config import START_URL, MAX_REQUESTS
from src.extractor import extract_and_tag

class NovoxCrawler:
    def __init__(self):
        self.crawler = PlaywrightCrawler(max_requests_per_crawl=MAX_REQUESTS, headless=True)
        self.output_file = "scraped_data_output.jsonl"
        with open(self.output_file, "w", encoding="utf-8") as f:
            pass

    def setup_routes(self):
        @self.crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            url = context.request.url
            await context.page.wait_for_load_state("networkidle")
            html = await context.page.content()
            processed_page = extract_and_tag(url, html)
            
            if processed_page:
                print(f"✅ [{processed_page['role'].upper()}] -> {url}")
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(processed_page) + "\n")
            await context.enqueue_links()

    async def start(self):
        self.setup_routes()
        print(f"🚀 Starting Playwright crawl at: {START_URL}")
        await self.crawler.run([START_URL])
        print(f"\n🎉 Done! Check '{self.output_file}'.")