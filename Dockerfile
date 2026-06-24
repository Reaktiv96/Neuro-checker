FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаём папки для данных (будут перемонтированы через volumes)
RUN mkdir -p backend/data backend/logs

EXPOSE 5001

CMD ["gunicorn", \
     "--workers", "2", \
     "--bind", "0.0.0.0:5001", \
     "--timeout", "120", \
     "--access-logfile", "backend/logs/access.log", \
     "--error-logfile", "backend/logs/error.log", \
     "backend.app:app"]
