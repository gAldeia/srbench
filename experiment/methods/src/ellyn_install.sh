#!/bin/bash
#install ellyn

# remove directory if it exists
if [ -d "ellyn" ]; then 
    rm -rf ellyn
fi

git clone  https://github.com/cavalab/ellyn

cd ellyn
# fix version
git checkout cdff25b2851d942db1cdb2a6796ea61c41396c7c

python setup.py install
