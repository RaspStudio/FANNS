FROM myapp-base

RUN apt-get update && apt-get install -y iproute2
WORKDIR /app
COPY . .
ENV HF_ENDPOINT=https://hf-mirror.com
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]