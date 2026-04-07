EXP_NAME="MK-M2Diffuser-Goal-Reach"

NUM_GPUS=$1

if [ -z "$NUM_GPUS" ]; then
    echo "Usage: ./train.sh <num_gpus>"
    exit 1
fi

GPUS="["
for ((i=0; i<NUM_GPUS; i++)); do
    if [ $i -gt 0 ]; then
        GPUS+=","
    fi
    GPUS+="$i"
done
GPUS+="]"

echo "Launching training on GPUs: ${GPUS}"

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                gpus="${GPUS}" \
                diffuser=ddpm \
                diffuser.loss_type=l2 \
                diffuser.timesteps=50 \
                model=m2diffuser_mk \
                model.use_position_embedding=true \
                task=mk_m2diffuser_goal_reach \
                task.train.num_epochs=2000 \