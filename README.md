# Simulator for AttAcc
This repository includes Python-based simulator designed to analyze the transformer-based generation model (TbGM) inference in a heterogeneous system consisting of an xPU and an Attention Accelerator (AttAcc). 
AttAcc is an accelerator for the attention layer of TbGM, which consists of an HBM-based processing-in-memory (PIM) structure.
In simulating an xPU and AttAcc system, the simulator outputs the performance and energy usage of the xPU, while the behavior of AttAcc is simulated using a properly modified [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2).
We set the memory device of AttAcc in Ramulator2 to HBM3 and implemented AttAcc\_bank, AttAcc\_BG, and AttAcc\_buffer, which represent AttAcc deploying processing units per bank, per bank group, or per pseudo-channel (on the buffer die), respectively.
For more details of AttAcc, please check the [paper](https://dl.acm.org/doi/10.1145/3620665.3640422) **AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference** published at [ASPLOS 2024](https://www.asplos-conference.org/asplos2024).

 
## Prerequisites
- Python
- cmake, g++, and clang++ (for building Ramulator2)

AttAcc simulator is tested under the following system.

* OS: Ubuntu 22.04.3 LTS (Kernel 6.1.45)
* Compiler: g++ version 12.3.0
* python 3.8.8

We use a similar build system (CMake) as original Ramulator 2.0, which automatically downloads following external libraries.
- [argparse](https://github.com/p-ranav/argparse)
- [spdlog](https://github.com/gabime/spdlog)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)


## How to install
1. Build Ramulator2
```bash
$ bash set_pim_ramulator.sh 
$ cd ramulator2
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp ramulator2 ../ramulator2
$ cd ../../
```

## How to run

## Details of the Ramulator for AttAcc
### How to Run
1. Generate PIM command traces for the Transformer-based Generative Model.
```bash
$ cd ramulator2
$ cd trace_gen
$ python ./trace_gen/sm_trace_attacc_bank.py -n 1 -m 1024 -k 1024
--> generate gemv_attacc_bank.trace
$ python ./trace_gen/sm_trace_attacc_bank.py -n 1 -m 1024 -k 1024 -p 4 
--> generate gemv_attacc_bank.trace
```
p는 packing factor 
n x k x m 이 n x k/p x m 이 된다고 대충 가정했음


2. Run Ramulator-
```bash
$ ./ramulator2 -f sm_attacc_bank.yaml 
-->attacc HBM-PIM 의 HBM3_5.2Gbps gem_attacc_bank.trace 실행
$ ./ramulator2 -f sm_packed_attacc_bank.yaml
-->attacc HBM-PIM 의 HBM3_LUT_5.2Gbps_ gem_attacc_bank_packed.trace 실행
```

### 수정법
/src/dram/impl/hbm3-pim
에서 표시해 놓은 파라미터 적절히 고치기!
MAC commands (`nCCDAB`).


## 추가 분석
log/attacc/ --> sm_attacc_bank.yaml의 실행 로그
log/attacc-LUT/ --> sm_packed_attacc_bank.yaml의 실행 로그