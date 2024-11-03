# Deeploy

![CI](https://github.com/pulp-platform/Deeploy/actions/workflows/CI.yml/badge.svg?branch=devel)
![Docker](https://github.com/pulp-platform/Deeploy/actions/workflows/BuildDocker.yml/badge.svg)
[![GitHub last commit](https://img.shields.io/github/last-commit/pulp-platform/Deeploy)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![](https://img.shields.io/badge/Provided_by_PULP_Platform-24AF4B)

Deeploy is an ONNX-to-C compiler that generates low-level optimized C Code for multi-cluster, heterogeneous SoCs. Its goal is to enable configurable deployment flows from a bottom-up compiler perspective, modeling target hardware in a fine-grained and modular manner.

Deeploy is developed as part of the PULP project, a joint effort between ETH Zurich and the University of Bologna.

## License

Unless specified otherwise in the respective file headers, all code checked into this repository is made available under a permissive license. All software sources and tool scripts are licensed under Apache 2.0, except for files contained in the `scripts` directory, which are licensed under the MIT license, and files contained in the `DeeployTest/Tests`directory, which are licensed under the [Creative Commons Attribution-NoDerivates 4.0 International](https://creativecommons.org/licenses/by-nd/4.0) license (CC BY-ND 4.0).

## Getting started

Installing Deeploy is as simple as running:
```
pip install -e .
```
However, to run the code generated by Deeploy on a certain target, you need the toolchains and the simulators associated with this platform.

We provide a Docker container where Deeploy works Out-of-the-Box (*i.e.* with all the dependencies pre-installed). To pull the docker image, run:
```
docker pull ghcr.io/pulp-platform/deeploy:main
```
Then you can start the container in interactive mode with:
```
docker run -it ghcr.io/pulp-platform/deeploy:main
```
From the container, clone Deeploy, its submodules, and install the package with:
```
git clone https://github.com/pulp-platform/Deeploy.git && cd Deeploy
git submodule update --init --recursive
pip install -e .
```
Congratulations, you installed Deeploy and its dependencies! Now, to test your installation let's run one simple test on each platform with the following commands:
```
cd DeeployTest && source /app/install/pulp-sdk/configs/siracusa.sh
python testRunner_generic.py -t Tests/Adder
python testRunner_cortexm.py -t Tests/Adder
python testRunner_mempool.py -t Tests/Adder
python testRunner_siracusa.py -t Tests/Adder --cores=8
```

You can find the ONNX file in `DeeployTest/Tests/Adder`, to visualize it, you can use [Netron](https://netron.app/). You can also find the generated code for the platform X in `TEST_X` in `DeeployTest` and you should notice that the generated code for the `Adder` test is very simple. However, this gets more complex when you add tiling. Let's generate the code for a single layer but using tiling this time:
```
python testRunner_tiled_siracusa.py -t Tests/testMatMul --cores=8 --l1=16000
``` 
Now you can open the generated code in `DeeployTest/TEST_SIRACUSA/Tests/testMatMul/Network.c` and see how we executed a tiled layer.

## Supported Platforms

- **Generic CPU:**
- **CortexM Processors:**
    - Simulators: [QEMU](https://www.qemu.org/)
- **MemPool extended with ITA:**
    - Hardware: [Mempool paper](https://arxiv.org/abs/2303.17742), [ITA paper](https://arxiv.org/abs/2307.03493)
    - Simulators: [Banshee](https://github.com/pulp-platform/banshee)
- **Siracusa:**
    - Hardware: [Siracusa paper](https://arxiv.org/abs/2312.14750)
    - Simulators: [GVSOC](https://github.com/gvsoc/gvsoc)

## Documentation

To build the documentation, simply run:
```
make docs
```
Then open `docs/_build/html/index.html` for more extensive documentation & getting-started guides.

## Publications

### ESWEEK 2024: Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers
```
@article{scherer2024deeploy,
  title={Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers},
  author={Scherer, Moritz and Macan, Luka and Jung, Victor and Wiese, Philip and Bompani, Luca and Burrello, Alessio and Conti, Francesco and Benini, Luca},
  journal={arXiv preprint arXiv:2408.04413},
  year={2024}
}
```
