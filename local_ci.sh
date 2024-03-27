SUBNAME=$1
SUBFOLDER="algorithms/$1"
echo "testing $SUBFOLDER"
# SUBFOLDER=official_competitors/$SUBNAME 
SUBENV=srbench-$SUBNAME 
# update base env
# mamba env update -n srbench -f environment.yml 

# install method
cd $SUBFOLDER
pwd
echo "Installing dependencies for ${SUBNAME}"
echo "........................................"
echo "Copying base environment"
echo "........................................"
conda create --name $SUBENV --clone srbench
if [ -e environment.yml ] ; then 
    echo "Installing conda dependencies"
    echo "........................................"
    conda env update -n $SUBENV -f environment.yml
fi
if [ -e requirements.txt ] ; then 
    echo "Installing pip dependencies"
    echo "........................................"
    conda run -n $SUBENV pip install -r requirements.txt
fi

eval "$(conda shell.bash hook)"
conda init bash
conda activate $SUBENV
if test -f "install.sh" ; then
echo "running install.sh..."
echo "........................................"
bash install.sh
else
echo "::warning::No install.sh file found in ${SUBFOLDER}. Assuming the method is a conda package specified in environment.yml."
fi

# Copy files and environment
echo "Copying files and environment to experiment/methods ..."
echo "........................................"
cd ../../
mkdir -p experiment/methods/$SUBNAME
cp $SUBFOLDER/regressor.py experiment/methods/$SUBNAME/
cp $SUBFOLDER/metadata.yml experiment/methods/$SUBNAME/
touch experiment/methods/$SUBNAME/__init__.py

# export env
echo "Exporting environment"
conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml

# Test Method
cd experiment
pwd
ls
echo "activating conda env $SUBENV..."
echo "........................................"
conda activate $SUBENV 
conda env list 
conda info 
python -m pytest -v test_algorithm.py --ml $SUBNAME

# Store Competitor
# cd ..
# rsync -avz --exclude=".git" submission/$SUBNAME official_competitors/
# rm -rf submission/$SUBNAME

# install environment (you need to create  one for each algorithm youre gonna run)
# bash local_ci.sh <name of model>

# Run experiments --------------------------------------------------------------
# https://cavalab.org/srbench/user-guide/#reproducing-the-experiment

# cd experiments
# conda activate environment?

# locally
# python analyze.py ../datasets/pmlb/datasets/ -n_trials 30 -results ../results_blackbox -time_limit 00:01:00 --local

# slurm
# conda activate srbench-feat (it can be any feat, as long as installed from my branch)
python analyze.py ../datasets/pmlb/datasets/ -n_trials 10 -results ../results_blackbox -time_limit 48:00 \
                                             -max_samples 10000 --slurm -q 'bch-compute'

# ground truth experiments (With brush, I need to remember to remove some operators)
# submit the ground-truth dataset experiment. 

for data in "../datasets/pmlb/datasets/strogatz_" "../datasets/pmlb/datasets/feynman_" ; do
    for TN in 0 0.001 0.01 0.1; do
        python analyze.py \
            $data"*" \
            -results ../results_sym_data \
            -target_noise $TN \
            -sym_data \
            -n_trials 10 \
            -m 8192 \
            -max_samples 10000 \
            -time_limit 9:00 \
            -job_limit 1000 \
            --slurm \
            -q 'bch-compute'
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done

# assess the ground-truth models that were produced using sympy
for data in "../datasets/pmlb/datasets/strogatz_" "../datasets/pmlb/datasets/feynman_" ; do
    for TN in 0 0.001 0.01 0.1; do
        python analyze.py \
            -script assess_symbolic_model \
            $data"*" \
            -results ../results_sym_data \
            -target_noise $TN \
            -sym_data \
            -n_trials 10 \
            -m 8192 \
            -time_limit 1:00 \
            -job_limit 1000 \
            --slurm \
            -q 'bch-compute'
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done