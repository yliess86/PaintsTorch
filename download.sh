##!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "$GREEN [download.sh] Creating 'res/model' folder $NC"
if [ ! -d res/model ]; then
  mkdir res/model
fi

echo "$GREEN [download.sh] Downloading models $NC"
filename="res/model/models.tar.xz"
file_id="1bPj7WxT9myNkoIH_vdCWTi8ZtzFqyf8R"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${filename} $url
tar -xf ${filename} -C res/model

echo "$GREEN [download.sh] Removing temp files $NC"
rm ${filename}
rm ./cookie.txt
