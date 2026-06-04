import json
import asyncio
import random
from datetime import timedelta
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
            ignore_http_error_status_codes=[401, 403, 429],
            request_handler_timeout=timedelta(minutes=3),
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
            
            # WAF/Firewall Interceptor: Prevent rapid-retry loops if StackCDN rate-limits Playwright
            if context.response and context.response.status in [401, 403, 429]:
                status = context.response.status
                print(f"⚠️ CDN Rate Limit (Status {status}). Cooling down for 60 seconds to reset firewall...")
                await asyncio.sleep(60)
                raise Exception(f"Retrying after CDN cool-down (Status {status})")
            
            # Enqueue links before processing
            await context.enqueue_links()
            
            # Playwright executes JS, solving the StackCDN Cloudflare-style challenges natively!
            await context.page.wait_for_load_state('networkidle')
            html_content = await context.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            processed_page = extract_and_tag(url, soup)
            
            if processed_page:
                print(f"✅ [{processed_page['role'].upper()}] -> {url}")
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(processed_page) + "\n")
            
            # Keep the randomized human jitter to avoid being blocked by StackCDN.
            # Increased to 5-8 seconds because the CDN rate limits after ~50 rapid requests
            sleep_time = random.uniform(5.0, 8.0) 
            await asyncio.sleep(sleep_time)

    async def start(self):
        self.setup_routes()
        print(f"🚀 Starting Playwright crawl at: {START_URL}")
        await self.crawler.run([START_URL])
        print(f"\n🎉 Done! Check '{self.output_file}'.")