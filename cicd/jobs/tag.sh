#!/bin/bash


# CURRENT_BRANCH=$(cat .git/HEAD | awk -F '/' '{print $NF}')

COMMIT_MSG=$(git log --pretty='format:%Creset%s' --no-merges -1)
git tag -a v$(python setup.py --version) -m "${COMMIT_MSG}" && git push origin --tags
