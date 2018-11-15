# Add kaggle.json?
kaggle competitions download -c titanic -p ./input
kaggle kernels pull -p ./tasks vprovv/sample-kernel
jupyter nbconvert --to notebook --execute ./tasks/sample-kernel.ipynb --output ./tasks/sample-kernel.ipynb
kaggle competitions submit -c titanic -f ./tasks/result.csv -m "Sample"