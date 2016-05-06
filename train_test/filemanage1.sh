#!/bin/zsh

# ファイルを移動させる用

FROM_PATH='data/apple_1_crop/'

TO_X_PATH='data_train/train/x/'
TO_Y_PATH='data_train/train/y/'


for FILE in ${FROM_PATH}*_crop.png
do
  echo ${FILE}
  cp ${FILE} ${TO_X_PATH} || echo "failed ${FILE}"
done


for FILE in ${FROM_PATH}*_depthcrop.png
do
  echo ${FILE}
  cp ${FILE} ${TO_Y_PATH} || echo "failed ${FILE}"
done
