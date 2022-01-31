#download dataset PMEmo2019
[ ! -d "data/raw/PMEmo2019" ] && echo "Downloading dataset" && gdown --id 1ea_6CDtguFSQO35aDimu4utlnLGFP5qf -O "data/raw/PMEmo2019.zip" && unzip "data/raw/PMEmo2019.zip" -d "data/raw" && rm "data/raw/PMEmo2019.zip" && rm -dR "data/raw/__MACOSX"

