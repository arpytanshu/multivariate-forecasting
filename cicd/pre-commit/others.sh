#!/bin/bash
if [[ ${FINAL+set} = set ]]; then
    echo "Running Tests"
    sh cicd/jobs/test.sh
    echo "Beautifying Code"
    sh cicd/jobs/beautify.sh
    echo "Checking Examples"
    sh cicd/jobs/check_examples.sh
    git add examples/*.ipynb
else
        echo "bypassing pre-commit hook"
fi