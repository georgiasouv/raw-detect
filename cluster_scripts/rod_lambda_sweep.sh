#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --array=0-7
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs_rod/rod_sweep_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs_rod/rod_sweep_%A_%a.err

# ---- map array index -> (config arm, lambda) ----
CONFIGS=(synergy_coupled_rod synergy_uncoupled_rod)
LAMBDAS=(0.005 0.01 0.05 0.1)
ARM=$(( SLURM_ARRAY_TASK_ID % 2 ))          # 0=coupled, 1=uncoupled
LIDX=$(( SLURM_ARRAY_TASK_ID / 2 ))         # 0..3 -> lambda index
CFG=${CONFIGS[$ARM]}
LAMBDA=${LAMBDAS[$LIDX]}
RUN_NAME="rod_${CFG#synergy_}_lam${LAMBDA}"

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

echo "=== Kerberos auth ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
klist
(
while true; do
    sleep 14400
    kinit -R -c FILE:/tmp/krb5cc_$(id -u) 2>/dev/null || \
        kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass 2>/dev/null
done
) &
KRENEW_PID=$!

echo "=== Trigger CIFS mount ==="
ls /cifs/Shares/WMGData/ > /dev/null 2>&1
ls /cifs/Shares/Raw_Bayer_Datasets/ > /dev/null 2>&1

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_MODE=offline

cd /networkhome/WMGDS/souval_g/raw-detect
export PYTHONPATH="$(pwd):${PYTHONPATH}"

WORKDIR=/networkhome/WMGDS/souval_g/raw-detect/work_dirs/rod/${RUN_NAME}
echo "=== Starting ${RUN_NAME} (arm=${CFG} lambda=${LAMBDA}) ==="
mim train mmdet configs/methods/${CFG}.py \
    --launcher none \
    --work-dir ${WORKDIR} \
    --cfg-options \
        custom_hooks.0.target=${LAMBDA} \
        visualizer.vis_backends.1.init_kwargs.name=${RUN_NAME} \
        visualizer.vis_backends.1.init_kwargs.group=rod_lambda_sweep

echo "=== syncing W&B ==="
wandb sync ${WORKDIR}/*/vis_data/wandb/offline-run-* 2>/dev/null
kill $KRENEW_PID 2>/dev/null
echo "=== ${RUN_NAME} finished ==="
