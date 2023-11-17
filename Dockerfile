FROM python:3.10-slim as build

RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.6.1

RUN pip install --upgrade pip \
    && pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN python -m venv /app/venv \
    && . /app/venv/bin/activate \
    && poetry install


FROM python:3.10-slim as prod
COPY --from=build /app/venv /app/venv/

ENV PATH /app/venv/bin:$PATH

COPY . ./

CMD ["uvicorn", "crowdstop.main_server:app", "--host", "0.0.0.0", "--port", "8000"]