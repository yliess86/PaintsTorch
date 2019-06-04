##!/usr/bin/env bash

echo "[download.sh] Creating 'res/model' folder"
if [ ! -d res/model ]; then
  mkdir res/model
fi

echo "[download.sh] Downloading models"
filename="res/model/models.tar.xz"
file_id="1bPj7WxT9myNkoIH_vdCWTi8ZtzFqyf8R"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${filename} $url
tar -xf ${filename} -C res/model

echo "[download.sh] Removing temp files"
rm ${filename}
rm ./cookie.txt
