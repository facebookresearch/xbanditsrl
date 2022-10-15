#!/bin/bash

for DATASET in wheel
do
    HORIZON=500000
    LR=0.001
    if [[ $DATASET == *"covertype"* ]]; then
        HORIZON=1000000
        LR=0.0001
    fi
    echo "#!/bin/bash" > run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlegreedy domain='${DATASET}'.yaml epsilon_decay=cbrt check_glrt=True forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=5 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run1 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run1/.slurm &' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrllinucb domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=5 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run2 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run2/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=bsrlegreedy epsilon_decay=cbrt check_glrt=false forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run3 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run3/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=bsrllinucb epsilon_decay=none check_glrt=false forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=1 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run4 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run4/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlegreedy domain='${DATASET}'.yaml epsilon_decay=cbrt check_glrt=True forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=5 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run5 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run5/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrllinucb domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=5 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run6 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run6/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m domain='${DATASET}'.yaml algo=bsrlts epsilon_decay=none check_glrt=false forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_weak=0 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run13 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run13/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlts domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_weak=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=5 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run14 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run14/.slurm & ' >> run_netexp_${DATASET}.sh

    echo 'python runner_exp.py -m algo=bsrlts domain='${DATASET}'.yaml epsilon_decay=none check_glrt=true forced_exploration_decay=none layers=5000,\"100,100\",\"50,50,10\",\"50,50,50\",\"50,50,50,50,10\",\"50,50,50,50,50\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=5 use_maxnorm=False device="cpu" lr='${LR}' hydra.sweep.dir=wheelablation/'${DATASET}'_5010/run15 hydra.launcher.submitit_folder=wheelablation/'${DATASET}'_5010/run15/.slurm & ' >> run_netexp_${DATASET}.sh

    chmod +x run_netexp_${DATASET}.sh
    echo "written run_netexp_${DATASET}.sh"
done