python -m t5.models.mesh_transformer_main \
--module_import="tasks.all_all" \
--model_dir="models/small" \
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
--gin_param="utils.run.mesh_devices = ['gpu:0']" \
--gin_param="MIXTURE_NAME = 'korsmr'" \
--gin_param="SentencePieceVocabulary.extra_ids=100" \
--gin_param="utils.run.batch_size=('tokens_per_batch', 2560)" \
--gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 512" \
--gin_param="eval_checkpoint_step = 'all'" \
--gin_param="run.dataset_split = 'validation'" \
--gin_file="models/small/operative_config.gin" \
--gin_file="gins/cnn_dailymail_v002.gin" \
--gin_file="gins/dataset.gin" \
--gin_file="gins/beam_search.gin" \
--gin_file="gins/eval.gin"