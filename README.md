# Fair-Vision

This repository consists the source coude for the project fairness in vision conducted at ETH SRI lab.

## Data prepration

First, put the CelebA data inside ``./data/celeba/``. Then run the following scripts to get the LMDB dataset for Celeba. The data will be saved in ``./data/celeba/celeba_lmdb``.

```
python scripts/convert_celeba_lmdb.py --split train
python scripts/convert_celeba_lmdb.py --split valid
python scripts/convert_celeba_lmdb.py --split test
```

## Training the VAE

For training the VAE, you can use the ```train_vae()``` function inside ``vae.py``. ``vae.py`` also contains a number of different functinos for visualizing the vae outputs. The model should be saved in ``./final-models/``

## Training the encoder

To train the base encoder, run
```
python3 train_encoder.py --base true
```
To train the encoder with LCIFR, run 
```
python3 train_encoder.py --base false
```

## Training the encoder

To train the base classifier, run
```
python3 train_classifier.py --robust false
```
To train the classifier with adversarial training, run
```
python3 train_classifier.py --robust true
```
## Producing the certification resluts

To get the certification results for the base model as well as the fair model, run 
```
python3 certify.py --robust false
python3 certify.py --robust true
```

The whole end to end pipeline can be run using
```
sh run.sh
```

## Changing the experiments
The experiment configurations can be modifier using ``metadata.json``.