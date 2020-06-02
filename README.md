# Learning Mutational Semantics

### Data

You can download the relevant datasets using the commands
```bash
wget http://cb.csail.mit.edu/cb/viral-mutation/data.tar.gz
tar xvf data.tar.gz
```
within the same directory as this repository.

### Dependencies

The major Python package requirements and their tested versions are in [requirements.txt](requirements.txt).

Our experiments were run with Python version 3.7.4 on Ubuntu 18.04.

### Experiments

#### Headlines

Training data on news headlines can be done with the command
```bash
python bin/headlines.py bilstm \
    --dim 512 \
    --train \
    --test \
    > headlines_bilstm512_train.log 2>&1
```

Loading a trained model from a checkpoint and generating news headlines according to CSCS can be done with the command
```bash
python bin/headlines.py bilstm \
    --dim 512 \
    --checkpoint MODEL_FNAME
    --semantics \
    > headlines_bilstm512_semantics.log 2>&1
```
where `MODEL_FNAME` is a path to the pretrained model checkpoint.

#### Flu

Training the model on flu HA sequences can be done with the command
```bash
python bin/flu.py bilstm \
    --dim 512 \
    --train \
    --test \
    > flu_bilstm512_train.log 2>&1
```

Flu semantic embedding UMAPs and log files with statistics can be generated with the command
```bash
python bin/flu.py bilstm \
    --dim 512 \
    --checkpoint MODEL_FNAME \
    --embed \
    > flu_bilstm512_embed.log 2>&1
```
where `MODEL_FNAME` is a path to the pretrained model checkpoint.

Single-residue escape prediction can be done with the command
```bash
python bin/flu.py bilstm \
    --dim 512 \
    --checkpoint MODEL_FNAME \
    --semantics \
    > flu_bilstm512_semantics.log 2>&1
```

Combinatorial fitness experiments and grammaticality can be done with the command
```bash
python bin/flu.py bilstm \
    --dim 512 \
    --checkpoint MODEL_FNAME \
    --combfit \
    > flu_bilstm512_combfit.log 2>&1
```

#### HIV

Training the model on HIV HA sequences can be done with the command
```bash
python bin/hiv.py bilstm \
    --dim 512 \
    --train \
    --test \
    > hiv_bilstm512_train.log 2>&1
```

HIV semantic embedding UMAPs and log files with statistics can be generated with the command
```bash
python bin/hiv.py bilstm \
    --dim 512 \
    --checkpoint MODEL_FNAME \
    --embed \
    > hiv_bilstm512_embed.log 2>&1
```
where `MODEL_FNAME` is a path to the pretrained model checkpoint.

Single-residue escape prediction can be done with the command
```bash
python bin/hiv.py bilstm \
    --dim 512 \
    --checkpoint MODEL_FNAME \
    --semantics \
    > hiv_bilstm512_semantics.log 2>&1
```
