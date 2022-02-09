FROM silverlogic/python3.6

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir /mer-music-emotion-recognition
WORKDIR /mer-music-emotion-recognition

RUN apt-get update

RUN pip install --no-cache-dir --upgrade pip pipenv
COPY Pipfile ./
RUN pipenv install --deploy --ignore-pipfile