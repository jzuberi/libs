from functools import wraps
from .logger import get_logger

logger = get_logger("llm.engine")

def log_engine_call(mode_name: str):
    """
    Decorator for logging engine mode calls.
    Logs:
      - mode name
      - prompt
      - raw backend output
      - cleaned output
      - parsed JSON
      - final schema
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "debug", False):
                return func(self, *args, **kwargs)

            logger.debug(f"=== MODE: {mode_name} ===")

            # Log input arguments
            logger.debug(f"ARGS: {args}")
            logger.debug(f"KWARGS: {kwargs}")

            result = func(self, *args, **kwargs)

            # Log final schema
            logger.debug(f"RESULT: {result}")

            logger.debug(f"=== END MODE: {mode_name} ===\n")
            return result

        return wrapper

    return decorator
