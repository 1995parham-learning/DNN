#!/bin/bash
# In The Name of God
# ========================================

url="https://storage.googleapis.com/tensorflow-serving-apt"
src="stable tensorflow-model-server tensorflow-model-server-universal"

echo "deb $url $src" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl "$url/tensorflow-serving.release.pub.gpg" | sudo apt-key add -
sudo apt update -q && sudo apt-get install -y tensorflow-model-server