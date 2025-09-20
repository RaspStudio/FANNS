FROM myapp-base

WORKDIR /app
COPY . .
ENV HF_ENDPOINT=https://hf-mirror.com


EXPOSE 8080

CMD ["python", "run_server.py"]