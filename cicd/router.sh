#!/bin/bash
CURRENT_BRANCH=$(cat .git/HEAD | awk -F '/' '{print $NF}')
COMMAND_MSG=$1

if [ ${CURRENT_BRANCH} = "dev" ];then
  bash cicd/${COMMAND_MSG}/dev.sh
elif [ ${CURRENT_BRANCH} = "master" ];then
  bash cicd/${COMMAND_MSG}/master.sh
else
  bash cicd/${COMMAND_MSG}/others.sh
fi
