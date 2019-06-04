#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "$GREEN [download.sh] $NC Creating 'res/model' folder"
if [ ! -d res/model ]; then
  mkdir res/model
fi

echo "$GREEN [download.sh] $NC Downloading models"
url="https://www.dropbox.com/s/n6tj7s08acbg285/models.tar.xz?dl=0"
filename="res/model/models.tar.xz"
wget -O ${filename} $url
tar -xf ${filename} -C res/model

echo "$GREEN [download.sh] $NC Removing temp files"
rm ${filename}
