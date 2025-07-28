# GOWalker

## Preliminaries
The following tools are required to reproduce our results.
* `C++>=9.4.0`: compiler supporting C++17.
* `cuda>=12.2`: support cuda complier
* `wget`: download the datasets.

## Build
1.clone our code  repository
```sh
git clone https://github.com/Teamb507/GOWalker.git
```
2.download dataset
```sh
cd GOWalker
mkdir dataset
cd script
./download_twitter.sh
./download_friendster.sh
./download_uk_union.sh
```
3.make our GOWalker,support two algorithm:Node2vec and SOPR
```sh
cd GOWalker
make
```
## Experiment
1.We provide some scripts for running each experiment with the exact same configuration used in the paper. 
In the dir `test`,the scripts support some experiments in our paper 
* `test/experiment1`: (figure 10)support the experiment to evaluate the efficiency of GOWalker by  comparing it against three state-of-the-art graph processing systems CGgraph, LightTraffic,and SOWalker, across five datasets.
* `test/experiment2`: (figure 13)support the experiment to evaluate the effectiveness of GOWalker’s CPU-GPU pipelining mechanism



