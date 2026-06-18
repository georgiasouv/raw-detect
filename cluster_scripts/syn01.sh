#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs/Syn01_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs/Syn01_%j.err

# ============================================================
#  Syn01  =  freezing study, variant 'frozen'  (configs/methods/real_Syn01_frozen.py)
# ============================================================

# -- Environment --------------------------------------------------
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

# -- Kerberos authentication --------------------------------------
echo "=== Attempting Kerberos authentication ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
echo "kinit exit status: $?"
klist
echo "=== Starting Kerberos renewal loop ==="
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
echo "Kerberos renewal loop started (PID: $KRENEW_PID)"

# -- Trigger CIFS mount -------------------------------------------
echo "=== Triggering CIFS mount ==="
ls /cifs/Shares/WMGData/ > /dev/null 2>&1
ls /cifs/Shares/Raw_Bayer_Datasets/ > /dev/null 2>&1

# -- W&B (headless node: log offline, sync after) -----------------
export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_MODE=offline

# -- Training -----------------------------------------------------
echo "=== Starting Syn01 (variant_frozen) ==="
cd /networkhome/WMGDS/souval_g/raw-detect
export PYTHONPATH="$(pwd):${PYTHONPATH}"

mim train mmdet configs/methods/real_Syn01_frozen.py \
    --launcher none \
    --work-dir /networkhome/WMGDS/souval_g/raw-detect/work_dirs/Syn01_frozen \
    --cfg-options \
        visualizer.vis_backends.1.init_kwargs.name=Syn01_frozen \
        visualizer.vis_backends.1.init_kwargs.group=freezing_sweep

# -- Sync the offline W&B run up to the dashboard -----------------
echo "=== syncing W&B offline run ==="
wandb sync /networkhome/WMGDS/souval_g/raw-detect/work_dirs/Syn01_frozen/*/vis_data/wandb/offline-run-* 2>/dev/null

# -- Cleanup ------------------------------------------------------
kill $KRENEW_PID 2>/dev/null
echo "=== Syn01 (variant_frozen) finished ==="