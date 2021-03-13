local_port=8888

docker run \
    --gpus all --rm  \
    -p $local_port:8888 \
    -v $PWD/trainer:/trainer \
    -v $PWD/trainer/data:/trainer/data \
    trainer &

sleep 1

docker exec -it $(docker ps -q --filter ancestor=trainer) bash
