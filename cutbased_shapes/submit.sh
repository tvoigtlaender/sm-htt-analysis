###############################
# Submit shape jobs to condor #
###############################
# create error log out folder if not existens
mkdir -p error log out

## source LCG Stack and submit the job
WORKDIR=${PWD}

if uname -a | grep -E 'el7' -q
then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_96/x86_64-centos7-gcc8-opt/setup.sh
    condor_submit ${WORKDIR}/cutbased_shapes/produce_shapes_cc7.jdl
elif uname -a | grep -E 'el6' -q
then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-slc6-gcc8-opt/setup.sh
    condor_submit ${WORKDIR}/cutbased_shapes/produce_shapes_slc6.jdl
else
    echo "Maschine unknown, i don't know what to do !"
fi
