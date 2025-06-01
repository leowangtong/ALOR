# How to install downstream datasets


We suggest putting all datasets under the same folder (say `data`)  to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```
# Task-specific data

data/
|–– semi_aves/
|–– aircraft/
|–– oxford_pets/
|–– stanford_cars/
|–– food101/

#Retrieved data

data/
|–– semi_aves/
    |–– retrieved_data/
|–– aircraft/
    |–– retrieved_data/
|–– oxford_pets/
    |–– retrieved_data/
|–– stanford_cars/
    |–– retrieved_data/
|–– food101/
    |–– retrieved_data/
```

Below we provide instructions to prepare the downstream datasets used in our experiments. Please refer to [RETRIEVAL.md](https://github.com/tian1327/SWAT/blob/master/retrieval/RETRIEVAL.md) for instructions on how to set up the retrieved datasets and put the retrieved data into data/[semi_aves, aircraft, oxford_pets, stanford_cars, food101]/retrieved_data/

Datasets list:

- [Semi-Aves](#semi-aves)

- [Aircraft](#fgvcaircraft)

- [OxfordPets](#oxfordpets)

- [StanfordCars](#stanfordcars)

- [Food101](#food101)



The instructions to prepare each dataset are detailed below. 
To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets.


### Semi-Aves

- Create a folder named `semi_aves/` under `data`.
- Download data from the [official repository](https://github.com/cvl-umass/semi-inat-2020) or following the `wget` commands below
```bash
cd data/semi_aves/

# train_val data
wget https://drive.google.com/uc?id=1xsgOcEWKG9CszNNT_EXN3YB1OLPYNbf8 

# test
wget https://drive.google.com/uc?id=1OVEA2lNJnYM5zxh3W_o_Q6K5lsNmJ9Hy

# unzip
tar -xzf *.gz
```
- The annotations are extracted from the [official annotation json files](https://github.com/cvl-umass/semi-inat-2020). We have reformatted labels and provided to you as `ltrain.txt`, `ltrain+val.txt`,`val.txt` and `test.txt` in the `TFS/data/semi_aves/` folder.
  
The directory structure should look like:

```
semi-aves/
|–– trainval_images
|–– test
```

### Aircraft
- Create a folder named `aircraft/` under `data`.
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
```
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `fgvc-aircraft-2013b/data/` to `data/aircraft`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `TFS/data/aircraft/` folder.

The directory structure should look like:
```
aircraft/
|–– fgvc-aircraft-2013b/
    |–– data/
```


### OxfordPets
- Create a folder named `oxford_pets/` under `data`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `TFS/data/oxford_pets/` folder.

The directory structure should look like:
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `data`.
- In case the following link breaks, download dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).
- Download `car_devkit.tgz`
```bash
wget https://github.com/pytorch/vision/files/11644847/car_devkit.tgz
tar -xzvf car_devkit.tgz
```
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).


The directory structure should look like
```
stanford_cars/
|–– cars_test\
|–– cars_annos.mat
|–– cars_train\
|–– split_zhou_StanfordCars.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `data` resulting in a folder, rename it as `data/food101/`.
```bash
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
```
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food101/
|–– images/
|–– license_agreement.txt   
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

---

## Acknowledgment
- We have already prepared the `train.txt`, `val.txt` and `test.txt` files for each dataset, using the same splits as in [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) and [SWAT](https://github.com/tian1327/SWAT/blob/master/DATASETS.md). We sincerely thank the authors for open-sourcing their projects.
