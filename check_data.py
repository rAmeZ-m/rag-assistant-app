import os
from pypdf import PdfReader

folder = "data/raw"
total_pages = 0

for name in sorted(os.listdir(folder)):
    try:
        reader = PdfReader(os.path.join(folder, name))
        pages = len(reader.pages)
        chars = sum(len(p.extract_text() or "") for p in reader.pages)
        total_pages += pages
        status = "OK" if chars > 500 else "PROBLEM"
        print(f"{name} | pages={pages} | chars={chars} | {status}")
    except Exception as e:
        print(f"{name} | FAILED | {e}")

print("Total pages:", total_pages)