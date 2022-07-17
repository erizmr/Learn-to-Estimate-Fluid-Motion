#!/bin/bash
mkdir -p pretrained_model
cd pretrained_model
fileid="1EOEN6zpTpYcbIieaov4D93rItH0fjZJl"
filename="predictor_unpwc_plain.pth"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}

fileid="1uWBXhwqreCLcxOhtGZBe_FMKpjirU7sT"
filename="corrector.pth"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}

cd ..