# QDNNS
Query Difficulty Driven Nearest Neighbor Search On Heterogeneous Machines

## Requirements

This project needs the following libraries:

1. LightGBM: https://github.com/microsoft/LightGBM
2. OpenMP: https://www.openmp.org/
3. BLAS: 

LightGBM install

```bash
https://github.com/microsoft/LightGBM.git
cd LightGBM
mkdir build && cd build
cmake ..
make -j4
```

Then add LightGBM/lib_lightgbm.so to the LD_LIBRARY_PATH.

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Train Router

First, we need groud truth for the vector database.

```bash
./build/gt_cpu wikipedia.base wikipedia.query
```

Then, build the HNSW graph index.

```bash
./build/hnsw_build wikipedia.base wikipedia.query 32 1000 0
```


Use [lightgbm_combine.py](./script/lightgbm_combine.py) to train the router.

## QDNNS

You can use [run.sh](./script/run.sh) to run the QDNNS, Qonly, EarlyStop and random algorithm.

## Acknowledgements

Thanks for the following repositories: 

1. [Faiss-GPU](https://github.com/facebookresearch/faiss-gpu)
2. [ANNS](https://github.com/PUITAR/ANNS)

The GPU-based exact KNNS is based on 1, and the CPU-based HNSW is based on 2.

## TO FIX

malloc_consolidate(): unaligned fastbin chunk detected
Aborted (core dumped)