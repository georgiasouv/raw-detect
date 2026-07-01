#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs/synergy_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-detect/cluster_scripts/logs/synergy_eval_%j.err

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

echo "=== Kerberos auth ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
echo "kinit exit: $?"
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

cd /networkhome/WMGDS/souval_g/raw-detect
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo ""
echo "############### COUPLED (Syn04) ###############"
python tools/synergy_eval.py configs/methods/synergy_coupled.py \
    work_dirs/Syn04_coupled/best_coco_bbox_mAP_epoch_*.pth \
    --out-dir work_dirs/synergy/Syn04 --tag Syn04_coupled \
    --max-train 1000 --max-val 851

echo ""
echo "############### UNCOUPLED (Syn05) #############"
python tools/synergy_eval.py configs/methods/synergy_uncoupled.py \
    work_dirs/Syn05_uncoupled/best_coco_bbox_mAP_epoch_*.pth \
    --out-dir work_dirs/synergy/Syn05 --tag Syn05_uncoupled \
    --max-train 1000 --max-val 851

kill $KRENEW_PID 2>/dev/null
echo ""
echo "=== synergy eval finished -- grep the two Spearman lines above ==="
