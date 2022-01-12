
export TASK_NAME=all_all
export TPU_NAME=your_tpu_address
export TPU_SIZE=your_tpu_size

python -m t5.models.mesh_transformer_main \
  --module_import="${TASK_NAME}" \
  --tpu="${TPU_NAME}" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '$TPU_SIZE'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --model_dir="models/base/" \
  --gin_file="models/small/operative_config.gin" \
  --gin_file="dataset.gin" \
  --gin_param="MIXTURE_NAME = 'korsmr'" \
  --gin_param="utils.run.save_checkpoints_steps=1000" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 196608)" \
  --gin_param="utils.run.train_steps=756700" \
  --gin_param="SentencePieceVocabulary.extra_ids=100" \


