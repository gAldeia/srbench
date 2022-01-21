#!/bin/bash

PYTHON_SITE=${CONDA_PREFIX}/lib/python`pkg-config --modversion python3`/site-packages

## aria-csv
mkdir -p ${CONDA_PREFIX}/include/aria-csv
wget https://raw.githubusercontent.com/AriaFallah/csv-parser/master/parser.hpp -O ${CONDA_PREFIX}/include/aria-csv/parser.hpp

## vectorclass
git clone https://github.com/vectorclass/version2.git vectorclass
mkdir -p ${CONDA_PREFIX}/include/vectorclass
cp vectorclass/*.h ${CONDA_PREFIX}/include/vectorclass/
rm -rf vectorclass
cat > ${CONDA_PREFIX}/lib/pkgconfig/vectorclass.pc << EOF
prefix=${CONDA_PREFIX}/include/vectorclass
includedir=${CONDA_PREFIX}/include/vectorclass

Name: Vectorclass
Description: C++ class library for using the Single Instruction Multiple Data (SIMD) instructions to improve performance on modern microprocessors with the x86 or x86/64 instruction set.
Version: 2.01.04
Cflags: -I${CONDA_PREFIX}/include/vectorclass
EOF

## vstat
git clone https://github.com/heal-research/vstat.git
pushd vstat
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf vstat

## pratt-parser
git clone https://github.com/foolnotion/pratt-parser-calculator.git
pushd pratt-parser-calculator
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf pratt-parser-calculator

## fast-float
git clone https://github.com/fastfloat/fast_float.git
pushd fast_float
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTLOAT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf fast_float

## span-lite
git clone https://github.com/martinmoene/span-lite.git
pushd span-lite
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPAN_LITE_OPT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf span-lite

## robin_hood
git clone https://github.com/martinus/robin-hood-hashing.git
pushd robin-hood-hashing
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRH_STANDALONE_PROJECT=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf robin-hood-hashing

# operon
git clone https://github.com/heal-research/operon.git
pushd operon
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon
cmake --install build
popd
rm -rf operon

## pyoperon
git clone https://github.com/heal-research/pyoperon.git
pushd pyoperon
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PYTHON_SITE}
cmake --build build -j -t pyoperon_pyoperon
cmake --install build
popd
rm -rf pyoperon
