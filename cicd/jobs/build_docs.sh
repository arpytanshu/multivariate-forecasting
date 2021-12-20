#Remove previously generated docs
rm -rf docs/source/examples
rm -rf docs/source/generated_api
rm docs/source/README.rst

#Copy examples
make --directory ./docs copy-examples

#Generate API documentation
make --directory ./docs generate

#Generate README
make --directory ./docs readme

#Build documentation
make --directory ./docs html

git add .
