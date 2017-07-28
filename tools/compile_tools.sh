#!/bin/bash
# 1. Getting SPTK-3.10

echo "compiling SPTK..."
(
    cd SPTK-3.10;
    ./configure --prefix=$PWD/build;
    make;
    make install
)

# 2. Getting WORLD

echo "compiling World..."
(
    cd World;
    make
    cd examples/analysis_synthesis;
    make
)

# 3. Copy binaries

SPTK_BIN_DIR=bin/SPTK-3.10
WORLD_BIN_DIR=bin/World

mkdir -p bin
mkdir -p $SPTK_BIN_DIR
mkdir -p $WORLD_BIN_DIR

cp SPTK-3.10/build/bin/* $SPTK_BIN_DIR/
cp World/build/analysis $WORLD_BIN_DIR/
cp World/build/synthesis $WORLD_BIN_DIR/

if [[ ! -f ${SPTK_BIN_DIR}/x2x ]]; then
    echo "Error installing SPTK tools"
    exit 1
elif [[ ! -f ${WORLD_BIN_DIR}/analysis ]]; then
    echo "Error installing WORLD tools"
    exit 1
else
    echo "All tools successfully compiled!!"
fi
