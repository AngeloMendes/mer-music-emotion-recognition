FROM silverlogic/python3.6

RUN mkdir /mer-music-emotion-recognition
WORKDIR /mer-music-emotion-recognition

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="$WORKDIR/venv/bin:$PATH"


RUN apt-get update

RUN pip install --no-cache-dir --upgrade pip pipenv

#COPY . ./
COPY Pipfile ./
RUN pipenv install --deploy --ignore-pipfile