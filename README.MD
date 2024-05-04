# Translations of this Dude's good neural network tutorials to python3
I translated 2 neural network tutorials from python 2 into python 3 and updated the pytorch tutorial using chatGPT.

## Installation
I used `pyenv` to install python 3.11:
```bash
brew update && brew install pyenv
pyenv install 3.11
cd <into this directory>
pyenv local 3.11 # use python 3.11 in this directory
python3 --version # double check
python3 -m venv .env
source .env/bin/activate
pip install requirements.txt 
```

## Directory structure

## toy_nn
Corresponds to this [simple neural network tutorial](https://iamtrask.github.io/2015/07/12/basic-python-network/). Start here. 

## gradient_decent
This should hopefully fill your gaps on [gradient decent](https://iamtrask.github.io/2015/07/27/python-network-part2/). The titles of the code are the same name as the heading in the tutorial. 

## pytorch
The updates to his [pytorch tutorial](https://iamtrask.github.io/2017/01/15/pytorch-tutorial/). I couldn't translate the pytorch code 1:1, but chatGPT gives the same output, but with differently looking code. `pytorch/ff_nn_with_pytorch.py` will not run, but `ff_nn_with_pytorch_chat_gpt_translation.py` is the updated version. 

Both of the `ff_nny_with_pytorch...` files reference the **Putting it all together** section. 

One thing that I noticed is that running the chatGPT version with the same number of iterations (2400) as the original resulted in an error the functionally did not change:
```
Error: 0.25448256731033325
Error: 0.2500939667224884
Error: 0.25007420778274536
Error: 0.2500563859939575
Error: 0.25003987550735474
Error: 0.2500244677066803
Error: 0.2500099539756775
Error: 0.24999597668647766
Error: 0.24998240172863007
Error: 0.24996905028820038
Error: 0.2499556988477707
Error: 0.24994222819805145
```

But running with the chatGPT suggested number of iterations (60 000), resulted in approaching low error

### Chat GPT output:
```
Error: 0.25003957748413086
Error: 0.029267750680446625
Error: 0.0029548280872404575
Error: 0.0014186393236741424
Error: 0.0009169571567326784
Error: 0.0006725460407324135
```

### the author's
```
Error:0.2521711587905884
Error:0.2500123083591461
Error:0.249952495098114
Error:0.24984735250473022
Error:0.2495250701904297
Error:0.2475520819425583
Error:0.22693687677383423
Error:0.13267411291599274
Error:0.04083901643753052
Error:0.016316475346684456
Error:0.008736669085919857
Error:0.005575092509388924
```
