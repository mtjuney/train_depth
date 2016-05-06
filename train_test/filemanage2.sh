#!/bin/zsh

# ファイルのリネーム用

FILE_PATH='data_train/train/y/'


for FILE in ${FILE_PATH}*
do
  echo ${FILE}
  OLD_NAME=${FILE}
  NEW_NAME=`echo ${FILE} | sed -e "s/depthcrop/crop/"`
  mv ${OLD_NAME} ${NEW_NAME} || echo "failed ${FILE}"
done
