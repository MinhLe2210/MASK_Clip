FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf,required=false \
--mount=type=cache,target=/root/.cache/pip \
pip install \
--default-timeout=100 \
--retries=3 \
-r requirements.txt

COPY . .

EXPOSE 8002

CMD ["uvicorn", "pipeline_server:app", "--host", "0.0.0.0", "--port", "8002"]
