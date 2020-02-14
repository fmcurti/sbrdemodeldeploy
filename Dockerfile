FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --upgrade -r requirements.txt

COPY ./app /app
