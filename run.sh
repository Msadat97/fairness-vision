# train the base encoder
python3 train_encoder.py --base true

# train the lcifr encoder
python3 train_encoder.py --base false

# training the base classifier
python3 train_classifier.py --robust false

# training the robust classifier
python3 train_classifier.py --robust true

# running the certification part
python3 certify.py --robust false
python3 certify.py --robust true

