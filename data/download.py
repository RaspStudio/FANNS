import os 


## iNaturalist
os.system("wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz")
os.system("wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz")
os.system("wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz")
os.system("wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/inat2018_locations.zip")
os.system("wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/test2018.tar.gz")


## ImageNet
os.system("wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar")


## MS MARCO
os.system("wget https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/test-00000-of-00001.parquet?download=true")
os.system("wget https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet?download=true")
os.system("wget https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/validation-00000-of-00001.parquet?download=true")


## Sentiment
os.system("wget --no-check-certificate http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")

