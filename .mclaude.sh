#!/bin/bash

#
# Code to run at the start of Claude container
#

if [[ ! -d ${HOME}/.venv/tinymoe ]]; then
    uv venv ${HOME}/.venv/tinymoe
    source ${HOME}/.venv/tinymoe/bin/activate
    uv sync --extra cpu --active
fi

http_pid=$(ps aux | grep -v grep | grep service/flyte-binary-http | awk '{ print $2 }')
grpc_pid=$(ps aux | grep -v grep | grep service/flyte-binary-grpc | awk '{ print $2 }')
minio_pid=$(ps aux | grep -v grep | grep service/minio | awk '{ print $2 }')

if [[ -z ${http_pid} ]]; then
    kubectl --namespace flyte port-forward service/flyte-binary-http 8088:8088 > /dev/null 2>&1 &
    http_pid=$!
fi

if [[ -z ${grpc_pid} ]]; then
    kubectl --namespace flyte port-forward service/flyte-binary-grpc 8089:8089 > /dev/null 2>&1 &
    grpc_pid=$!
fi

if [[ -z ${minio_pid} ]]; then
    kubectl --namespace flyte port-forward service/minio 9000:9000 9090:9090 > /dev/null 2>&1 &
    minio_pid=$!
fi
