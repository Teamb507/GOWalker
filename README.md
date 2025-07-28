# GOwalker
Code for our [VLDB paper](论文网址)
and [arXiv report](链接).
## Preliminaries
The following tools are required to reproduce our results.
* `C++>=9.4.0`: compiler supporting C++17.
* `cuda>=12.2`: support cuda complier
* `wget`: download the datasets.

## Build
1.clone our code  repository
```sh
git clone 
```
2.download dataset
```sh
cd GOwalker
mkdir dataset
cd script
./download_twitter.sh
./download_friendster.sh
./download_uk_union.sh
```
3.make our GOwalker,support two algorithm:Node2vec and SOPR
```sh
cd GOwalker
make
```
## Experiment
1.We provide some scripts for running each experiment with the exact same configuration used in the paper. 
In the dir `test`,the scripts support some experiments in our paper 
* `test/test1.sh`: support the experiment to  
* `test/test2.sh`: support the experiment to  
* `test/test3.sh`: support the experiment to  



