FROM python:3.12-slim

WORKDIR /app

# نسخة torch الخفيفة (CPU) بدل نسخة الـ GPU الضخمة
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نزّل موديل الـ embeddings وقت الـ build عشان التشغيل ميحتاجش نت
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
COPY data ./data

ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]