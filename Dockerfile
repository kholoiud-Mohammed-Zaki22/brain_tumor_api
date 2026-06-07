FROM python:3.10-slim

# تثبيت التحديثات الأساسية إذا لزم الأمر
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود التطبيق وملف النموذج
# تأكد أن ملف النموذج موجود في نفس المجلد محلياً قبل الرفع
COPY main.py .
COPY brain_tumors_classifier.keras .

# ضبط المنفذ (Railway يمرر المنفذ عبر متغير البيئة PORT)
EXPOSE 8000

# تشغيل التطبيق
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
