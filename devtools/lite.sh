#!/bin/bash
# delete all the Test and unnecessary files
pushd .

rm -rf functional
rm -rf ipython
rm -rf arc/testing

find "$PWD" -path '*Test.py' -type f -delete

popd || exit
