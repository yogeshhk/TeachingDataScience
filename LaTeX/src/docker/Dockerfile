FROM python:3

WORKDIR /usr/src/app

COPY monalisa_noisy.jpg .
COPY nlm.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./nlm.py"]