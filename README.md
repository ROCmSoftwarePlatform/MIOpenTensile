MIOpenTensile provides host-callable interfaces to Tensile library.

MIOpenTensile supports one programming model: HIP.

MIOpenTensile is an open-source collaboration tool where external entities can submit source pull requests (PRs) for updates. These PRs are reviewed and approved by MIOpenTensile maintainers using standard open source practices. The sources, build system, and testing for MIOpenTensile are located in this github repository: https://github.com/ROCmSoftwarePlatform/MIOpenTensile 

The contents in the MIOpenTensile github repository were created and contributed by MIOpenTensile team.

To build MIOpenTensile, 
1. git clone https://github.com/ROCmSoftwarePlatform/MIOpenTensile -b master
2. cd MIOpenTensile
3. mkdir build; cd build
4. CXX=${ROCM_PATH}/bin/hipcc cmake ..
