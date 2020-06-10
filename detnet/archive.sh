#!/bin/bash

TAG=`git rev-parse --short=8 HEAD`
OUTFILE=../cervical_cancer-${TAG}.zip
echo $OUTFILE
git-archive-all $OUTFILE

