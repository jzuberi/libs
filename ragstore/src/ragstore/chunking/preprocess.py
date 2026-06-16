# ragstore/chunking/preprocess.py

def preprocess(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = "\n".join(line for line in text.split("\n") if line)
    return text.strip()
