# mer-music-emotion-recognition

Dependencies:
- Python 3.8
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Pipenv `pip install pipenv`

Run `docker build -t mer .` to build a docker image

Run `download_dataset.sh` to download PEMo2019 dataset

Create virtual environment `pipenv install && pipenv shell`

Run notebooks using virtualenv libs: `pipenv run jupyter notebook`, or config directly in IDE.

 docker exec -it mer cat Pipfile.lock > Pipfile.lock
 sudo docker container start mer
 sudo docker container run mer
 sudo docker container stop mer