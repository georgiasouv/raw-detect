#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs/Syn04_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs/Syn04_%j.err

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

echo "=== Attempting Kerberos authentication ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
echo "kinit exit status: $?"
klist
(
while true; do
    sleep 14400
    kinit -R -c FILE:/tmp/krb5cc_$(id -u) 2>/dev/null
    if [ $? -ne 0 ]; then
        kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass 2>/dev/null
    fi
done
) &
KRENEW_PID=$!

echo "=== Triggering CIFS mount ==="
ls /cifs/Shares/WMGData/ > /dev/null 2>&1
ls /cifs/Shares/Raw_Bayer_Datasets/ > /dev/null 2>&1

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_MODE=offline

echo "=== Starting Syn04 (synergy_coupled) ==="
cd /networkhome/WMGDS/souval_g/raw-detect
export PYTHONPATH="$(pwd):${PYTHONPATH}"

mim train mmdet configs/methods/synergy_coupled.py \
    --launcher none \
    --work-dir /networkhome/WMGDS/souval_g/raw-detect/work_dirs/Syn04_coupled \
    --cfg-options \
        visualizer.vis_backends.1.init_kwargs.name=Syn04_coupled \
        visualizer.vis_backends.1.init_kwargs.group=synergy

echo "=== syncing W&B offline run ==="
wandb sync /networkhome/WMGDS/souval_g/raw-detect/work_dirs/Syn04_coupled/*/vis_data/wandb/offline-run-* 2>/dev/null

kill $KRENEW_PID 2>/dev/null
echo "=== Syn04 (synergy_coupled) finished ==="
