#!/bin/bash
mkdir -p data
cd data
fileid="1R5oM1Wa_lrE_yWv5Hv0yljxvPYtJmIfx"
filename="DNS_turbulence_small.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
unzip -q DNS_turbulence_small.zip

cd ..