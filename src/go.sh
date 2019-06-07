
# rm doc2vecc
gcc doc2vecc.c -o doc2vecc -lm -pthread -O3 -march=native -funroll-loops

# this script trains on all the data (train/test/unsup), you could also remove the test documents from the learning of word/document representation
time ./doc2vecc -train ../data/alldata.txt         \
				-output ../data/docvectors.txt     \
				-word ../data/wordvectors.txt      \
				-cbow 1                            \
				-size 100                          \
				-window 10                         \
				-negative 5                        \
				-hs 0                              \
				-sample 0                          \
				-threads 8                         \
				-binary 0                          \
				-iter 20                           \
				-min-count 10                      \
				-test ../data/alldata.txt          \
				-sentence-sample 0.1               \
				-save-vocab ../data/alldata.vocab


