# rethink-audio-fsl
This repo contains the source code for the paper "Who calls the shots? Rethinking Few-Shot Learning for Audio." (WASPAA 2021)

**Table of Contents**
- [Setup](#setup)
- [Dataset](#dataset)
- [Experiment](#experiment)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Reference](#reference)
- [Citation](#citation)


## Setup
1. Clone the repo.

```
git clone git@github.com:wangyu/rethink-audio-fsl.git 
```
2. Create `conda` environment from the `environment.yml` file and activate it.
```
conda env create -f environment.yml
conda activate dfsl
```

## Dataset
Models in this work are trained on [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo), an open dataset of programmatically mixed audio clips with a controlled level of polyphony and signal-to-noise ratio. We use single-labeled clips from [FSD50K](https://zenodo.org/record/4060432#.YWyLAEbMIWo) as the source material for the foreground sound events and Brownian noise as the background to generate 281,039 10-second strongly-labeled soundscapes with [Scaper]. We refer this (intermediate) dataset of 10s soundscapes as FSD-MIX-SED. Each soundscape contains n events from n different sound classes where n is ranging from 1 to 5. We then extract 614,533 1s clips centered on each sound event in the soundscapes in FSD-MIX-SED to produce FSD-MIX-CLIPS. 

Due to the large size of the dataset, instead of releasing the raw audio files, we release the source material and soundscape annotations in JAMS format, which can be used to reproduce FSD-MIX-SED using [Scaper](https://github.com/justinsalamon). All clips in FSD-MIX-CLIPS are extracted from FSD-MIX-SED. Therefore, for FSD-MIX-CLIPS, instead of releasing duplicated audio content, we provide annotations that specify the filename in FSD-MIX-SED and the corresponding starting time (in second) of each 1-second clip.  

To reproduce FSD-MIX-SED:
1. Download source material and jams files from [Zenodo](https://zenodo.org/record/5574135#.YWyINEbMIWo).
2. Generate soundscapes from jams files by running:
```
python ./data/generate_soundscapes.py \
--annpath PATH-TO-FSD_MIX_SED.annotations \
--audiopath PATH-TO-FSD_MIX_SED.source \
--savepath PATH-TO-SAVE-OUTPUT
```
Note that the output is ~450GB with 281,039 audio files.

If you want to get the foreground material (FSD-MIX-SED.source) directly from FSD50K instead of downloading them, run
```
python ./data/preprocess_foreground_sounds.py \
--fsdpath PATH-TO-FSD50K \
--outpath PATH_TO_SAVE_OUTPUT
```

## Experiment
We provide source code to train the best performing embedding model (pretrained OpenL3 + FC) and three different few-shot methods to predict both base and novel class data.  

## Preprocessing
Once audio files are reproduced, we pre-compute OpenL3 embeddings of clips in FSD-MIX-CLIPS and save them by running
```
python get_openl3emb_and_filelist.py \
--annpath PATH-TO-FSD_MIX_CLIPS.annotations \
--audiopath PATH-TO-FSD_MIX_SED-AUDIO \
--savepath PATH_TO_SAVE_OUTPUT
```
This generates 614,533 `.pkl` files where each file contains an embedding. A set of filelists will also be saved under current folder.

## Training
- Training configuration can be specified using config files in `./config`
- Model checkpoints will be saved in the folder `./experiments`, and tensorboard data will be saved in the folder `./run`

### 1. Base classifier
First, to train the base classifier on base classes, run
```
python train.py --config openl3CosineClassifier --openl3
```

### 2. Few-shot weight generator for DFSL
Once the base model is trained, we can train the few-shot weight generator for DFSL by running
```
python train.py --config openl3CosineClassifierGenWeightAttN5 --openl3
```

By default, DFSL is trained with 5 support examples: `n=5`, to train DFSL with different `n`, run
```
# n=10
python train.py --config openl3CosineClassifierGenWeightAttN10 --openl3

# n=20
python train.py --config openl3CosineClassifierGenWeightAttN20 --openl3

# n=30
python train.py --config openl3CosineClassifierGenWeightAttN30 --openl3

```

## Evaluation
We evaluate the trained models on test data from both base and novel classes. For each novel class, we need to sample a support set. Run the command below to split the original filelist for test classes to `test_support_filelist.pkl` and `test_query_filelist.pkl`. 
```
python get_test_support_and_query.py
```
- Here we consider monophonic support examples with mixed(random) SNR. Code to run evaluation with polyphonic support examples with specific low/high SNR will be released soon. 

For evaluation, we compute features for both base and novel test data, then make predictions and compute metrics in a joint label space. The computed features, model predictions, and metrics will be saved in the folder `./experiments`. We consider 3 few-shot methods to predict novel classes. To test different number of support examples, set different `n_pos` in the following commands.

### 1. Prototype
```
# Extract embeddings of evaluation data and save them.
python save_features.py --config=openl3CosineClassifier --openl3

# Get and save model prediction, run this multiple time (niter) to count for random selection of novel examples.
python pred.py --config=openl3CosineClassifier --openl3 --niter 100 --n_base 59 --n_novel 15 --n_pos 5

# compute and save evaluation metrics based on model prediction
python metrics.py --config=audioset_pannCosineClassifier --openl3 --n_base 59 --n_novel 15 --n_pos 5
```

### 2. DFSL
```
# Extract embeddings of evaluation data and save them.
python save_features.py --config=openl3CosineClassifierGenWeightAttN5 --openl3

# Get and save model prediction, run this multiple time (niter) to count for random selection of novel examples.
python pred.py --config=openl3CosineClassifierGenWeightAttN5 --openl3 --niter 100 --n_base 59 --n_novel 15 --n_pos 5

# compute and save evaluation metrics based on model prediction
python metrics.py --config=audioset_pannCosineClassifierGenWeightAttN5 --openl3 --n_base 59 --n_novel 15 --n_pos 5
```

### 3. Logistic regression
Train a binary logistic regression model for each novel class. Note that we need to sample `n_neg` of examples from the base training data as the negative examples. Default `n_neg` is 100. We also did a hyperparameter search on `n_neg` based on the validation data while `n_pos` changing from 5 to 30:
- `n_pos=5, n_neg=100`
- `n_pos=10, n_neg=500`
- `n_pos=20, n_neg=1000`
- `n_pos=30, n_neg=5000`

```
# Extract embeddings of evaluation data and save them.
python save_features.py --config=openl3CosineClassifier --openl3

# Train binary logistic regression models, predict test data, and compute metrics
python logistic_regression.py --config=openl3CosineClassifier --openl3 --niter 10 --n_base 59 --n_novel 15 --n_pos 5 --n_neg 100
```

## Reference
This code is built upon the implementation from [FewShotWithoutForgetting](https://github.com/gidariss/FewShotWithoutForgetting)

## Citation
Please cite our paper if you find the code or dataset useful for your research.

Y. Wang, N. J. Bryan, J. Salamon, M. Cartwright, and J. P. Bello. "Who calls the shots? Rethinking Few-shot Learning for Audio", IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2021


