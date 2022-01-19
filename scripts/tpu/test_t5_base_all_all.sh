export TASK_NAME=all_all
export TPU_NAME=your_tpu_address
export TPU_SIZE=your_tpu_size

python -m t5.models.mesh_transformer_main \
--module_import="${TASK_NAME}" \
--tpu="${TPU_NAME}" \
--model_dir="models/base/" \
--gin_param="utils.tpu_mesh_shape.tpu_topology = '$TPU_SIZE'" \
--gin_param="MIXTURE_NAME = 'korsmr'" \
--gin_param="SentencePieceVocabulary.extra_ids=100" \
--gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
--gin_param="eval_checkpoint_step = 'all'"
--gin_param="run.dataset_split = 'validation'" \
--gin_file="models/base/operative_config.gin" \
--gin_file="gins/cnn_dailymail_v002.gin" \
--gin_file="gins/dataset.gin" \
--gin_file="gins/beam_search.gin" \
--gin_file="gins/eval.gin"