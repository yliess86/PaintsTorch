#!/usr/bin/env bash

nvidia-docker run \
  -it \
  --name Paintstorch \
  -v /home/yliess/Projects:/Projects \
  pytorch/pytorch:latest
