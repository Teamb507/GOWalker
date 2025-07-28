#the script supports the figure 9 of the paper and run the experiment to prove the overall Performance for GOwalker
# please make sure that the code has been compiled successfully!
cd ../..
make
# first test the Node2Vec
# Please replace the dataset path in the command with your own dataset path. Please replace the dataset path in the command with your own dataset path.
# the result of every experiment will be saved in the GOwalker/randgraph_metrics.txt
rm -f randgraph_metrics.txt
# if cannot find the dataset,please replace the dataset to all path
echo "start experiment to test all system performance of Node2vec"
./apps/Node2vec ./dataset/twitter/twitter.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 2 cpu_threads 32  p 0.5 q 2 blocksize 256  gpu cpu zero skip
./apps/Node2vec ./dataset/com-friendster/com-friendster.unDir.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 4 cpu_threads 32  p 0.5 q 2 blocksize 512  gpu cpu zero skip
./apps/Node2vec ./dataset/uk-union/uk.txt.unDir length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 8 cpu_threads 32  p 0.5 q 2 blocksize 1024  gpu cpu zero skip
./apps/Node2vec ./dataset/yahoo/yahoo.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 16 cpu_threads 32  p 0.5 q 2 blocksize 4096  gpu cpu zero skip
./apps/Node2vec ./dataset/k30/k30.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 16 cpu_threads 32  p 0.5 q 2 blocksize 4096  gpu cpu zero skip
python ./test/experiment1/ex1_extract.py
python ./test/experiment1/4-time-n2v.py

echo "start experiment to test all system performance of SOPR"
rm -f randgraph_metrics.txt
rm -f ./test/experiment1/result.csv
./apps/SOPR ./dataset/twitter/twitter.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 2 cpu_threads 32  p 0.5 q 2 blocksize 256  gpu cpu zero skip
./apps/SOPR ./dataset/com-friendster/com-friendster.unDir.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 4 cpu_threads 32  p 0.5 q 2 blocksize 512  gpu cpu zero skip
./apps/SOPR ./dataset/uk-union/uk.txt.unDir length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 8 cpu_threads 32  p 0.5 q 2 blocksize 1024  gpu cpu zero skip
./apps/SOPR ./dataset/yahoo/yahoo.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 16 cpu_threads 32  p 0.5 q 2 blocksize 4096  gpu cpu zero skip
./apps/SOPR ./dataset/k30/k30.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 16 cpu_threads 32  p 0.5 q 2 blocksize 4096  gpu cpu zero skip
python ./test/experiment1/ex1_extract.py
python ./test/experiment1/4-time-SOPR.py
