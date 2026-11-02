mkjob () 
{ 
    J=${QUEUE:?set QUEUE to the job queue folder}/x9_$1;
    mkdir -p $J;
    cp concept.py batch.py stream.json ../../v2/bench/world.py ../../v2/bench/physics.py $J/;
    cp $2 $J/specs.txt;
    printf '#!/bin/bash\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\npython3 -I batch.py specs.txt 4\nfor f in out/*.log; do echo == $f; tail -n 2 $f; done\n' > $J/job.sh;
    echo 80 > $J/TIMEOUT;
    touch $J/SUBMIT;
    echo submitted $J
}
