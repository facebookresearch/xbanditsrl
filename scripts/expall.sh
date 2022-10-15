#!/bin/bash

for DATASET in mushroom covertype magic statlog wheel
do
    HORIZON=500000
    LR=0.001
    if [[ $DATASET == *"covertype"* ]]; then
        HORIZON=1000000
        LR=0.0001
    fi
    echo "#!/bin/bash" > run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlegreedy domain='${DATASET}'.yaml epsilon_decay=cbrt,sqrt check_glrt=True forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run1 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run1/.slurm &' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrllinucb domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=0.1,1,2 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run2 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run2/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=bsrlegreedy epsilon_decay=cbrt,sqrt check_glrt=false forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run3 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run3/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=bsrllinucb epsilon_decay=none check_glrt=false forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=0.1,1,2 glrt_scale=1 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run4 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run4/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlegreedy domain='${DATASET}'.yaml epsilon_decay=cbrt,sqrt check_glrt=True forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run5 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run5/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrllinucb domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=0.1,1,2 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run6 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run6/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrligw domain='${DATASET}'.yaml epsilon_decay=cbrt,sqrt gamma_scale=1,10,50,100 check_glrt=True forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expbigw/'${DATASET}'_5010/run55 hydra.launcher.submitit_folder=expbigw/'${DATASET}'_5010/run55/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=igwexp refit_linear=false epsilon_decay=none forced_exploration_decay=none check_glrt=false gamma_exponent=sqrt,cbrt gamma_scale=1,10,50,100 forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run9 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run9/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=gradientucb epsilon_decay=none forced_exploration_decay=none check_glrt=false bonus_scale=0.1,1,2,5 layers=\"50,50,50,50,10\" horizon='${HORIZON}' seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run10 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run10/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=rffegreedy epsilon_decay=sqrt,cbrt forced_exploration_decay=none check_glrt=false bonus_scale=1 layers=100,300 horizon='${HORIZON}' seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run11 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run11/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=rfflinucb epsilon_decay=none forced_exploration_decay=none check_glrt=false bonus_scale=0.1,1 layers=100,300 horizon='${HORIZON}' seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run12 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run12/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=bsrlts epsilon_decay=none check_glrt=false forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=0 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=0.1,1,2 device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run13 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run13/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlts domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=0.1,1,2 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run14 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run14/.slurm & ' >> run_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlts domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=0.1,1,2 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=expall/'${DATASET}'_5010/run15 hydra.launcher.submitit_folder=expall/'${DATASET}'_5010/run15/.slurm & ' >> run_${DATASET}.sh

    chmod +x run_${DATASET}.sh
    echo "written run_${DATASET}.sh"
done