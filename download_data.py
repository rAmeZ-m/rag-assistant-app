import arxiv, os, time, requests

os.makedirs("data/raw", exist_ok=True)

topics = [
    "large language models",
    "computer vision deep learning",
    "reinforcement learning",
    "natural language processing",
    "machine learning survey",
]

client = arxiv.Client()
seen = set()

for topic in topics:
    search = arxiv.Search(query=topic, max_results=6, sort_by=arxiv.SortCriterion.Relevance)
    for r in client.results(search):
        pid = r.get_short_id().replace("/", "_")
        if pid in seen:
            continue
        seen.add(pid)
        resp = requests.get(r.pdf_url, timeout=60)
        resp.raise_for_status()
        with open(f"data/raw/{pid}.pdf", "wb") as f:
            f.write(resp.content)
        print(r.title)
        time.sleep(1)