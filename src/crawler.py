import json
import asyncio
import random
from crawlee import ConcurrencySettings
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext
from crawlee.http_clients import CurlImpersonateHttpClient
from src.config import START_URL, MAX_REQUESTS
from src.extractor import extract_and_tag

class NovoxCrawler:
    def __init__(self):
        self.crawler = BeautifulSoupCrawler(
            http_client=CurlImpersonateHttpClient(impersonate="chrome120"),
            max_requests_per_crawl=MAX_REQUESTS,
            ignore_http_error_status_codes=[401, 403, 429],
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
        async def request_handler(context: BeautifulSoupCrawlingContext) -> None:
            url = context.request.url
            status = context.http_response.status_code
            
            # WAF/Firewall Interceptor: Prevent rapid-retry loops if temporarily blocked
            if status in [401, 403, 429]:
                print(f"⚠️ WAF Blocked (Status {status}). Cooling down for 60 seconds to reset firewall...")
                await asyncio.sleep(60)
                raise Exception(f"Retrying after WAF cool-down (Status {status})")

            # CRITICAL: Enqueue links BEFORE decomposing the nav/header/footer tags
            await context.enqueue_links()
            
            processed_page = extract_and_tag(url, context.soup)
            
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
        print(f"🚀 Starting BeautifulSoup crawl at: {START_URL}")
        await self.crawler.run([START_URL])
        print(f"\n🎉 Done! Check '{self.output_file}'.")