#!/bin/bash

if [ ! -e ci/build.sh ]; then
  echo "Must run this from the root repo."
  exit 1
fi

docker -f ci/Dockerfile
