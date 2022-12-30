#!/bin/bash

# Script to reproduce results

#envs=(
#	"halfcheetah-random-v0"
#	"hopper-random-v0"
#	"walker2d-random-v0"
#	"halfcheetah-medium-v0"
#	"hopper-medium-v0"
#	"walker2d-medium-v0"
#	"halfcheetah-expert-v0"
#	"hopper-expert-v0"
#	"walker2d-expert-v0"
#	"halfcheetah-medium-expert-v0"
#	"hopper-medium-expert-v0"
#	"walker2d-medium-expert-v0"
#	"halfcheetah-medium-replay-v0"
#	"hopper-medium-replay-v0"
#	"walker2d-medium-replay-v0"
#	)

baseenvs=(
  "halfcheetah"
  "hopper"
  "walker2d"
)

extensions=(
  "random-v0"
  "medium-v0"
  "expert-v0"
  "medium-expert-v0"
  "medium-replay-v0"
)

for ((i=9;i<11;i+=1))
do
	for env in ${baseenvs[*]}
	do
    for extension in ${extensions[*]}
    do
      python3 main.py \
      --env $env \
      --extension $extension \
      --seed $i
    done
  done
done