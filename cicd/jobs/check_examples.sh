jupyter nbextension enable --py widgetsnbextension
cd examples
for FILE in  NbeatsAD.ipynb forecasting.ipynb 
do
	papermill $FILE $FILE
done
cd ../

git add .
# FFT-examples.ipynb RNN-examples.ipynb cliffts-intro.ipynb