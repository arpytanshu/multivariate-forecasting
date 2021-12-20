#!/bin/bash
# echo "Running Tests"
# sh cicd/jobs/test.sh
echo "Beautifying Code"
sh cicd/jobs/beautify.sh
# echo "Checking Examples"
# sh cicd/jobs/check_examples.sh
echo "Building Docs"
sh cicd/jobs/build_docs.sh
echo "Deploying Docs"
sh cicd/jobs/deploy_docs.sh
