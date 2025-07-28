#this script show the figure 12 "EffectivenessofCPU-GPUcooperativepipeline"
#node2vec
#compile the code to OPEN the GPU pipeline
cd ../..
make clean
make
rm -f randgraph_metrics.txt
# if cannot find the dataset,please replace the dataset to all path
echo "start experiment to test pipeline performance of Node2vec"
./apps/Node2vec ./dataset/twitter/twitter.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 2 cpu_threads 32  p 0.5 q 2 blocksize 256  gpu cpu zero skip
./apps/Node2vec ./dataset/com-friendster/com-friendster.unDir.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 4 cpu_threads 32  p 0.5 q 2 blocksize 512  gpu cpu zero skip
./apps/Node2vec ./dataset/uk-union/uk.txt.unDir length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 8 cpu_threads 32  p 0.5 q 2 blocksize 1024  gpu cpu zero skip
python ./test/experiment2/extract_total.py
#recompile the code to close the GPU pipeline
LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64::$LD_LIBRARY_PATH nvcc -I. -Xcompiler -fopenmp -w -D NO_pipeline -o apps/Node2vec apps/Node2vec.cu 
rm -f randgraph_metrics.txt
# if cannot find the dataset,please replace the dataset to all path
echo "start experiment to test no_pipeline performance of Node2vec"
./apps/Node2vec ./dataset/twitter/twitter.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 2 cpu_threads 32  p 0.5 q 2 blocksize 256  gpu cpu zero skip
./apps/Node2vec ./dataset/com-friendster/com-friendster.unDir.txt length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 4 cpu_threads 32  p 0.5 q 2 blocksize 512  gpu cpu zero skip
./apps/Node2vec ./dataset/uk-union/uk.txt.unDir length 80 walkpersource 2 nthreads 70 walk_batch 4096 memory 8 cpu_threads 32  p 0.5 q 2 blocksize 1024  gpu cpu zero skip
python ./test/experiment2/extract_nopipeline.py
##plot the result
python ./test/experiment2/n2v_plot.py

#SOPR
#compile the code to OPEN the GPU pipeline
make clean
make
rm -f randgraph_metrics.txt
echo "start experiment to test pipeline performance of SOPR"
./apps/SOPR ./dataset/twitter/twitter.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 2 cpu_threads 32  p 0.5 q 2 blocksize 256  gpu cpu zero skip
./apps/SOPR ./dataset/com-friendster/com-friendster.unDir.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 4 cpu_threads 32  p 0.5 q 2 blocksize 512  gpu cpu zero skip
./apps/SOPR ./dataset/uk-union/uk.txt.unDir length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 8 cpu_threads 32  p 0.5 q 2 blocksize 1024  gpu cpu zero skip
python ./test/experiment2/extract_total.py
#recompile the code to close the GPU pipeline
LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64::$LD_LIBRARY_PATH nvcc -I. -Xcompiler -fopenmp -w -D NO_pipeline -o apps/SOPR apps/SOPR.cu 
rm -f randgraph_metrics.txt
# if cannot find the dataset,please replace the dataset to all path
echo "start experiment to test no_pipeline performance of Node2vec"
./apps/SOPR ./dataset/twitter/twitter.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 2 cpu_threads 32  p 0.5 q 2 blocksize 256  gpu cpu zero skip
./apps/SOPR ./dataset/com-friendster/com-friendster.unDir.txt length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 4 cpu_threads 32  p 0.5 q 2 blocksize 512  gpu cpu zero skip
./apps/SOPR ./dataset/uk-union/uk.txt.unDir length 20 walkpersource 2 nthreads 70 walk_batch 4096 memory 8 cpu_threads 32  p 0.5 q 2 blocksize 1024  gpu cpu zero skip
python ./test/experiment2/extract_nopipeline.py
##plot the result
python ./test/experiment2/n2v_plot.py