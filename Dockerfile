FROM python :3.11- slim

WORKDIR /app
COPY . /app/

RUN apt-get update && apt-get install -y -no-install-recommends \
   build-essential \
   && rm -rf /var/lib/apt/list/*

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python","app.py" ]