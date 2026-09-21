import json, os, re
import arxiv

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "backend", "data", "titles.json")


def to_arxiv_id(filename):
    stem = filename[:-4]
    if re.match(r"^[a-z\-]+_\d{7}(v\d+)?$", stem):  # old-style ids like cmp-lg_9803002v1
        stem = stem.replace("_", "/", 1)
    return stem


def base(x):
    return re.sub(r"v\d+$", "", x)


files = sorted(f for f in os.listdir(RAW) if f.endswith(".pdf"))
by_base = {base(to_arxiv_id(f)): f for f in files}

client = arxiv.Client()
search = arxiv.Search(id_list=[to_arxiv_id(f) for f in files], max_results=len(files))

titles = {}
for r in client.results(search):
    fname = by_base.get(base(r.get_short_id()))
    if fname:
        titles[fname] = " ".join(r.title.split())

missing = [f for f in files if f not in titles]
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(titles, f, indent=2, ensure_ascii=False)
print(len(titles), "titles saved. Missing:", missing)