#!/bin/bash

# COMMIT_MSG=$(git log --pretty='format:%Creset%s' --no-merges -1)
# if [[ $COMMIT_MSG = \release* ]]
# then

# else
#   echo "Bumping Version: Skipped as the commit message doesn't start with release"
# fi


#script to bump up version
COMMIT_MSG=$(git log --pretty='format:%Creset%s' --no-merges -1)
#there is a bug in above line - it olny picks up last commit, not current commit. So it's good for post-merge call but not post comit

if [[ $COMMIT_MSG = \major* ]]
then
  echo "Bumping Version: Major"
  bump2version major --allow-dirty && git add .; git add .
elif [[ $COMMIT_MSG = \minor* ]]
then
  echo "Bumping Version: Minor"
  bump2version minor --allow-dirty && git add .; git add .
else
  echo "Bumping Version: Patch"
  bump2version minor --allow-dirty && git add .; git add .
fi

git add .
