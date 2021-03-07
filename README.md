# End-to-end deep representation learning for time series clustering

This is the code corresponding to the experiments conducted for the work
"End-to-end deep representation learning for time series clustering: a comparative study" 
(Baptiste Lafabregue, Jonathan Weber, Pierre Gançarki & Germain Forestier),
in submission

The results obtained for this paper on both archives are reported in the [paper_results/ folder](https://github.com/blafabregue/TimeSeriesDeepClustering/tree/main/paper_results)

## Datasets

The dataset used for the paper are available at : http://www.timeseriesclassification.com/ 
Both the univariate and the multivariate archive can be used. 

## Usage

### Install packages
You can use your favorite package manager (conda is recommended), create a new environment of python 3.8 or greater 
and use the packages listed in [requirements.txt](requirements.txt)
```sh
pip install -r requirements.txt
```

### For training on GPU
If you wish to train the networks on GPU you must install the tensorflow-gpu package. For example with pip:
```sh
pip install tensorflow-gpu
```
or 
```sh
conda install tensorflow-gpu
```
### Extract dataset to numpy format

First, to train the networks you need to convert them into .npy files. 
The CBF dataset is provided as an example. 
To do so you can use the `utils.py` script but you need first to change the two last line 
of the script [here](https://github.com/blafabregue/TimeSeriesDeepClustering/blob/1503c70053bbc8ec5ff34032c69e45099012c4ea/utils.py#L552). 
Note that this script is suited to extract data from .sktime files

### Train networks

To train networks and obtain the results you can use the following command:
```sh
python ucr_experiments.py --dataset <dataset_name> --itr <itreation_number> --architecture <network_architecture> --encoder_loss <enc> --clustering_loss IDEC --archives <archive_name> --nbneg <negative_exmaple_for_tiplet_loss> --hyper default_hyperparameters.json
```
Here is an example of the DRNN-rec-DEC combination on the CBF dataset
```sh
python ucr_experiments.py --dataset CBF --itr 1 --architecture dilated_cnn --encoder_loss reconstruction --clustering_loss DEC --archives UCRArchive_2018 --hyper default_hyperparameters.json
```
more informations are provided if you type directly `python ucr_experiments.py` in your prompt

### Results

The results are stored in two folders:

* ae_weights/<itreation_number>/<combination_name>/<dataset> 
will contain the logs (performance/loss evolution) and the saved weights of both 
the pretrained model (without clustering loss) and the final model, 
for the final model it will also save the representation learned of the train and test test 
,the clustering map and the centroids
* stats/<itreation_number> will contain the final statistics used to store the summarized 
stats used to evaluate methods. It will contain a .error file if an error occurred during the process

### Other computations
Baseline can be computed with following script:
```sh
python ucr_baseline.py --dataset <dataset_name> --itr <itreation_number> --archives <archive_name>
```
To combine already computed representations (i.e. for triplet compbined) it can be done with the following script (example with DCNN architecture):
```sh
python merge_trainers.py --dataset <dataset_name> --itr <itreation_number> --archives <archive_name> --prefix "dilated_cnn_tripletK" --suffix  "_None_0.0"
```
To use a reduction method on top of an already computed representation you can use:
```sh
python alternate_clustering.py --dataset <dataset_name> --itr <itreation_number> --architecture <network_architecture> --encoder_loss <enc> --clustering_loss IDEC --archives <archive_name> --nbneg <negative_exmaple_for_tiplet_loss>
```
For this last option the argument should be the semae as the one used to first compute the model. 
Note that the arguments are used to find load saved .npy files in ae_weights folder, 
so they should not have been used or renamed.