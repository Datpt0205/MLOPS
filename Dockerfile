# Base image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy thư mục src vào /app/src
COPY ./src /app/src

# *** QUAN TRỌNG: Copy model đã được chuẩn bị sẵn vào trong image ***
# Thư mục ./model_to_serve sẽ được tạo và chứa dữ liệu bởi job build-docker-image trong workflow CI/CD
COPY ./model_to_serve /app/model_to_serve

# Expose cổng
EXPOSE 8000

# Chạy bằng gunicorn cho production
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-", "api:app"]