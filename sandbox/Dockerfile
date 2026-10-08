FROM python:3.12-slim

RUN python -m pip install --no-cache-dir pytest==8.3.5

WORKDIR /workspace
USER 10001:10001

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTEST_ADDOPTS="-p no:cacheprovider"
