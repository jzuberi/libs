import concurrent.futures

class TimeoutRunnable:
    """
    Wraps any callable with a timeout.
    Returns either the callable's result (string) or a JSON-like error dict.
    """

    def __init__(self, func, timeout: int = 10):
        self.func = func
        self.timeout = timeout

    def __call__(self, *args, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                return '{"error": "timeout"}'
            except Exception as e:
                return f'{{"error": "backend failure: {str(e)}"}}'
