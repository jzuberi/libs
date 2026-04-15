# src/pensive/utils.py

def make_idea(key: str, **fields):
    return {"key": key, **fields}
