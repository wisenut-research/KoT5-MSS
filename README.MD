# KoT5-MSS
### KoT5 Media-Script-Summarization

## Setup
- 사전학습에 사용된 모델 버전은 t5==0.6.4, mesh_tensorflow==0.1.17
- 반드시 t5 모듈을 먼저 설치하고 rouge-score을 삭제해야함
```
pip install t5==0.6.4
pip install mesh_tensorflow==0.1.17
```
- 평가에는 형태소 토큰 단위 Rouge가 사용되었으며, Komoran 라이브러리를 통해 형태소 분석을 수행함
```
pip uninstall rouge-socere
pip install konlpy
```


## Data

### Download
- 아래 링크의 데이터셋을 내려받은 후 ./datas 폴더에서 압축해제
***(변경될 예정)***
  
| Data | Download | 
| --- | --- | 
| KBS 방송요약 | [link](https://storage.googleapis.com/nlp_bucket-1/KoT5_models/kbs_dialog_smr/datas.zip) | 

### Preprocessing
- 아래 경로의 스크립트 실행
```
#./scripts/data/all_all.sh
python ./datas/preprocessing.py --data_dir ./datas/ --domain all --output_length all
```

## Model Install
- 아래 링크의 모델을 내려받은 후 ./models 폴더에서 압축해제
***(변경될 예정)***

| model | Download | 
| --- | --- | 
| T5-Small, T5-Base, Vocab | [link](https://storage.googleapis.com/nlp_bucket-1/KoT5_models/kbs_dialog_smr/models.zip) |


## Train example
- 아래 경로의 스크립트 실행 \./scripts/gpu/train_t5_small_all_all.sh
```
python -m t5.models.mesh_transformer_main \
--module_import="tasks.all_all" \
--model_dir="models/small/" \
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
--gin_param="utils.run.mesh_devices = ['gpu:0']" \
--gin_param="MIXTURE_NAME = 'korsmr'" \
--gin_param="utils.run.train_steps=120000" \
--gin_param="SentencePieceVocabulary.extra_ids=100" \
--gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
--gin_param="utils.run.batch_size=('tokens_per_batch', 2560)" \
--gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 512" \
--gin_param="utils.run.save_checkpoints_steps=100" \
--gin_file="models/small/operative_config.gin"
--gin_file="gins/cnn_dailymail_v002.gin" \
--gin_file="gins/dataset.gin"
```

## Test Example
- 아래 경로의 스크립트 실행 \
./scripts/gpu/test_t5_small_all_all.sh
```
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
```



## License
**KoT5**는 [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 라이선스 하에 공개되어 있습니다. \
모델을 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 [LICENSE](./LICENSE) 파일에서 확인하실 수 있습니다.
