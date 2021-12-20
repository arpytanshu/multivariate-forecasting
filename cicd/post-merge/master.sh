#!/bin/bash
echo "Bumping Version"
sh cicd/jobs/bump.sh;
echo "Commiting Version Change"
COMMIT_MSG=$(git log --pretty='format:%Creset%s' --no-merges -1)
git add .
git commit -am "${COMMIT_MSG} -- version bump"
# echo "Releasing"
# git tag -a v$(python setup.py --version) -m "${COMMIT_MSG}" && git push origin --tags
