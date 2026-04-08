#!/bin/bash

# Build from root directory

cd ..

# Build image

registry="metalcycling"
name="tinymoe"
tag="latest"
image="${registry}/${name}:${tag}"

docker build --platform linux/aarch64 --tag ${image} --file image/Dockerfile .
docker tag ${image} localhost:5001/${name}:${tag}
docker push localhost:5001/${name}:${tag}
