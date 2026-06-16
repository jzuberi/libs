from datetime import datetime, timedelta
from .models import RetrievalIntent

def apply_time_postprocessing(intent: RetrievalIntent) -> RetrievalIntent:
    """
    Convert recency_days and time_bucket into canonical date_after/date_before.
    """

    now = datetime.utcnow()

    # -----------------------------------------
    # 1. Handle recency_days
    # -----------------------------------------
    if intent.recency_days is not None:
        delta = timedelta(days=intent.recency_days)
        cutoff = now - delta
        intent.date_after = cutoff.strftime("%Y-%m-%d")

    # -----------------------------------------
    # 2. Handle time_bucket
    # -----------------------------------------
    if intent.time_bucket:
        bucket = intent.time_bucket

        if bucket == "last_month":
            # Compute the first day of this month
            first_of_this_month = now.replace(day=1)
            # Last month ends at first day of this month
            end = first_of_this_month
            # Last month starts one month before
            start = (first_of_this_month.replace(day=1) - timedelta(days=1)).replace(day=1)

            intent.date_after = start.strftime("%Y-%m-%d")
            intent.date_before = end.strftime("%Y-%m-%d")

        elif bucket == "this_month":
            start = now.replace(day=1)
            intent.date_after = start.strftime("%Y-%m-%d")
            intent.date_before = None

        elif bucket == "this_week":
            start = now - timedelta(days=now.weekday())
            intent.date_after = start.strftime("%Y-%m-%d")
            intent.date_before = None

        elif bucket == "last_week":
            start = now - timedelta(days=now.weekday() + 7)
            end = start + timedelta(days=7)
            intent.date_after = start.strftime("%Y-%m-%d")
            intent.date_before = end.strftime("%Y-%m-%d")

        # Add more buckets as needed

    return intent
