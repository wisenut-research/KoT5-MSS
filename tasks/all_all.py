import functools
import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.evaluation import metrics as t5_metrics
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
import tensorflow as tf
import t5
from t5.data.utils import TextLineTask
from t5.data import preprocessors
from t5.data.utils import Feature

TaskRegistry = t5.data.TaskRegistry

#FILE PATH
file_list = ['datas/Training_all_all.tsv']
file_list2 = ['datas/Test_all_all.tsv']

##VOCAB PATH
vocab_model_path = 'models/sentencepiece.model'

vocab = SentencePieceVocabulary(vocab_model_path, extra_ids=100)
print("Vocab has a size of %d\n" %vocab.vocab_size)

corpus_path = {
    "train": file_list,
    "validation" : file_list2
}

def task_dataset_fn(split, shuffle_files=False):
  del shuffle_files

  ds = tf.data.TextLineDataset(corpus_path[split])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", "", "", ""],# 0이아닌 ""
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.map(lambda *ex: dict(zip(["output_length", "domain", "context", "summary"], ex)))
  return ds


def task_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs": normalize_text(ex["context"]),
            "targets": normalize_text(ex["summary"])
        }

    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.remove("korsmr")

t5.data.TaskRegistry.add(
    "korsmr",
    dataset_fn = task_dataset_fn,
    splits=["train", "validation"],
    text_preprocessor=task_preprocessor,
    output_features=t5.data.Feature(vocabulary=vocab, add_eos=True),
    metric_fns=[t5.evaluation.metrics.rouge]
)

