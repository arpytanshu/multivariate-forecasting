if [[ ${FINAL+set} = set ]]; then
        echo "Building Docs"
        sh cicd/jobs/build_docs.sh
        echo "Cleaning Changes post commit"
        git reset HEAD --hard
        git clean -fd
        
else
        echo "bypassing post-commit hook"
fi
