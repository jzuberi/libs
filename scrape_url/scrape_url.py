import sys, os, asyncio, atexit, threading
from queue import Queue
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

import time

# ============================================================
#  PLAYWRIGHT WORKER THREAD
# ============================================================

def robust_fetch(url, retries=3, delay=5.5, verbose=False):

    if(verbose):
        print(url)
    
    for attempt in range(retries):
        html = get_url_text(url, raw_only=True)
        if html:
            return html

        time.sleep(delay)

    # Final failure
    return None

class PlaywrightWorker:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.worker())

    async def worker(self):
        # Launch Playwright + persistent browser once
        self.pw = await async_playwright().start()

        user_data_dir = "playwright_profile"
        os.makedirs(user_data_dir, exist_ok=True)

        """
        self.context = await self.pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless="new",
            viewport={"width": 1280, "height": 800},
        )
        """

        self.context = await self.pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            args=[
                "--disable-gpu",             # reduce kernel_task load
                "--disable-dev-shm-usage",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-site-isolation-trials",
                "--renderer-process-limit=3",
                "--disable-accelerated-2d-canvas",
                "--disable-accelerated-video-decode",
                "--disable-accelerated-video-encode",
                "--disable-accelerated-mjpeg-decode",
            ],
            viewport={"width": 128, "height": 80},
        )




        self.page = await self.context.new_page()

        # Process tasks forever
        while True:
            func, args, kwargs = self.task_queue.get()
            if func is None:
                break  # shutdown signal

            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                result = ""

            self.result_queue.put(result)

        # Shutdown
        await self.page.close()
        await self.context.close()
        await self.pw.stop()

    def submit(self, func, *args, **kwargs):
        self.task_queue.put((func, args, kwargs))
        return self.result_queue.get()

    def shutdown(self):
        self.task_queue.put((None, None, None))
        self.thread.join()


# Global worker
# ============================================================
#  SINGLETON WORKER INSTANCE
# ============================================================

_worker_instance = None

def get_worker():
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = PlaywrightWorker()
    return _worker_instance

worker = get_worker()

# ============================================================
#  PUBLIC SYNC API
# ============================================================

def get_url_text(url, raw_only=False, verbose=False, timeout=60000):
    return worker.submit(fetch_url_with_playwright, url, raw_only, verbose, timeout)


# ============================================================
#  ASYNC PLAYWRIGHT FETCHER
# ============================================================

async def fetch_url_with_playwright(url, raw_only=False, verbose=False, timeout=60000):
    page = worker.page

    html = ""

    for attempt in range(2):
        try:
            await page.goto("about:blank")
            await page.goto(url, timeout=timeout)
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(5)
            html = await page.content()

            if html.strip():
                break

        except Exception as e:
            if verbose:
                print(f"[Attempt {attempt+1}] Error fetching {url}: {e}")
            html = ""

    if raw_only:
        return html

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)


# ============================================================
#  CLEAN SHUTDOWN
# ============================================================

@atexit.register
def shutdown_playwright():
    worker.shutdown()
