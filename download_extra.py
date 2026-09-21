import os, requests

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW, exist_ok=True)

for pid in ["1706.03762", "1810.04805"]:   # Attention Is All You Need + BERT
    r = requests.get(f"https://arxiv.org/pdf/{pid}", timeout=60)
    r.raise_for_status()
    with open(os.path.join(RAW, f"{pid}.pdf"), "wb") as f:
        f.write(r.content)
    print("saved", pid, "->", RAW)