#!/bin/bash

deactivate
pipenv run cxfreeze \
  -O \
  --compress \
  --target-dir=dist \
  --bin-includes="libffi.so" \
  --target-name="reading-script-generator" \
  src/cli.py
  
echo "compiled."
# copy to local apps folder
mkdir -p /home/mi/apps/reading-script-generator
cp dist/* -r /home/mi/apps/reading-script-generator
echo "deployed."

if [ $1 ]
then
  cd dist
  zip reading-script-generator-linux.zip ./ -r
  cd ..
  echo "zipped."
fi
