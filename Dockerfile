FROM python:3.11.9-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ADD ./src /app
RUN pip install -Ur requirements_fly.txt

FROM python:3.11.9-slim-bookworm

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY --from=builder /app .
ENV PATH="/opt/venv/bin:$PATH"

CMD sleep infinity