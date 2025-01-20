#!/bin/bash

set -x

# . /opt/intel/oneapi/setvars.sh
# for f in $(find /opt/intel/oneapi -name \*.cmake | grep -v /doc/ | grep -v /examples/ | xargs -L 1 dirname | sort | uniq); do
#         CMAKE_PREFIX_PATH=$f:${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}
# done
# export CMAKE_PREFIX_PATH

mkdir -p ./build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j $(grep processor /proc/cpuinfo | wc -l)

RESDIR=../results
mkdir -p $RESDIR

shellWidthParam=0.2;

sampleMin=0;
sampleMax=99;
##### Generic spin systems #####
NMin=6; NMax=10;
# NMin=6; NMax=14;
for N in $(seq $NMin $NMax); do
	mMin=1; mMax=$N;
	time ./PBC_TI/ShortRange_Spin/quasiETHmeasure_mBody_onGPU $N $N $mMax $mMin $sampleMin $sampleMax $shellWidthParam $RESDIR
done
# NMin=15; NMax=18;
# mMin=4; mMax=6;
# for m in $(seq $mMin $mMax); do
# 	time ./PBC_TI/ShortRange_Spin/quasiETHmeasure_mBody_onGPU $NMax $NMin $m $m $sampleMin $sampleMax $shellWidthParam $RESDIR
# done

##### Generic boson and fermion systems #####
ell=3; k=2; # 3-local and 2-body interactions
# NMin=6; NMax=8;
NMin=6; NMax=11;
mMin=1; mMax=3;
for N in $(seq $NMin $NMax); do
	L=$N;
	time ./PBC_TI/ShortRange_Boson/quasiETHmeasure_mBody_onGPU   $L $N $mMax $mMin $ell $k $sampleMin $sampleMax $shellWidthParam $RESDIR
	L=$((2 * N));
	time ./PBC_TI/ShortRange_Fermion/quasiETHmeasure_mBody_onGPU $L $N $mMax $mMin $ell $k $sampleMin $sampleMax $shellWidthParam $RESDIR
done
# mMim=4; mMax=6;
# for m in $(seq $mMin $mMax); do
# for N in $(seq $NMin $NMax); do
# 	L=$N;
# 	time ./PBC_TI/ShortRange_Boson/quasiETHmeasure_mBody_onGPU   $L $N $m $m $ell $k $sampleMin $sampleMax $shellWidthParam $RESDIR
# 	L=$((2 * N));
# 	time ./PBC_TI/ShortRange_Fermion/quasiETHmeasure_mBody_onGPU $L $N $m $m $ell $k $sampleMin $sampleMax $shellWidthParam $RESDIR
# done
# done

##### Mixed-field Ising model #####
Jx=1;
Bz=0.9045;
Bx=0.8090;
mMin=1;
# NMin=6; NMax=10;
NMin=6; NMax=14;
for N in $(seq $NMin $NMax); do
	mMax=$N;
	time ./PBC_TI/IsingModel/quasiETHmeasure_mBody_onGPU $N $mMax $mMin $Jx $Bz $Bx $shellWidthParam $RESDIR
done
# mMin=1;  mMax=6;
# NMin=15; NMax=18;
# for m in $(seq $mMin $mMax); do
# for N in $(seq $NMin $NMax); do
# 	time ./PBC_TI/IsingModel/quasiETHmeasure_mBody_onGPU $N $m $m $Jx $Bz $Bx $shellWidthParam $RESDIR
# done
# done


##### Bose-Hubbard model #####
t=1;
U=1;
# NMin=4; NMax=8;
NMin=4; NMax=11;
mMin=1; mMax=3;
for N in $(seq $NMin $NMax); do
	L=$N;
	time ./PBC_TI/BoseHubbard/quasiETHmeasure_mBody_onGPU $L $L $N $N $mMax $mMin $t $U 0 0 $shellWidthParam $RESDIR
done
# mMin=4; mMax=6;
# for N in $(seq $NMin $NMax); do
# 	L=$N;
# 	time ./PBC_TI/BoseHubbard/quasiETHmeasure_mBody_onGPU $L $L $N $N $mMax $mMin $t $U 0 0 $shellWidthParam $RESDIR
# done

##### spinless fermion model #####
J1=1;
U1=1;
J2=0.32;
U2=0.32;
mMin=1; mMax=3;
# NMin=4; NMax=8;
NMin=4; NMax=11;
for N in $(seq $NMin $NMax); do
	L=$((2 * N));
	time ./PBC_TI/FermiHubbard/quasiETHmeasure_mBody_onGPU $L $L $N $N $mMax $mMin $J1 $U1 $J2 $U2 $shellWidthParam $RESDIR
done
# mMin=4; mMax=6;
# for N in $(seq $NMin $NMax); do
# 	L=$((2 * N));
# 	time ./PBC_TI/FermiHubbard/quasiETHmeasure_mBody_onGPU  $L $L $N $N $mMax $mMin $J1 $U1 $J2 $U2 $shellWidthParam $RESDIR
# done