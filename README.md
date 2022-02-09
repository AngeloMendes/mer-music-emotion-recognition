# mer-music-emotion-recognition

Dependencies:
- [Python 3.7.12](https://www.python.org/downloads/release/python-3712/) 
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Pipenv `pip install pipenv`

[Install Python](https://opensource.com/article/20/4/install-python-linux):
1. `tar -xf Python-3.7.12.tgz`
2. `cd Python-3.7.12/`
3. `./configure`
4. `sudo make altinstall`

Ps.: If install result in error "ctypes not found" solution is [here](https://stackoverflow.com/a/66849415/6016300)

Run `docker build -t mer .` to build a docker image

Run `download_dataset.sh` to download PEMo2019 dataset

Create virtual environment `pipenv install && pipenv shell`

Run notebooks using virtualenv libs: `pipenv run jupyter notebook`, or config directly in IDE.

 docker exec -it mer cat Pipfile.lock > Pipfile.lock
 sudo docker container start mer
 sudo docker container run mer
 sudo docker container stop mer
 
docker exec -it mer pipenv install ...


