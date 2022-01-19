

import collections
import re
from absl import logging
from rouge_score import rouge_scorer
from rouge_score import scoring
import argparse

def rouge(targets, predictions, score_keys=None):
    """Computes rouge score.
    Args:
    targets: list of strings
    predictions: list of strings
    score_keys: list of strings with the keys to compute.
    Returns:
    dict with score_key: rouge score across all targets and predictions
    """

    if score_keys is None:
        score_keys = ["rouge1", "rouge2", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(score_keys)
        aggregator = scoring.BootstrapAggregator()

    for prediction, target in zip(predictions, targets):
        aggregator.add_scores(scorer.score(target=target, prediction=prediction))
        result = aggregator.aggregate()
        for key in score_keys:
            logging.info(
                "%s = %.2f, 95%% confidence [%.2f, %.2f]",
                key,
                result[key].mid.fmeasure*100,
                result[key].low.fmeasure*100,
                result[key].high.fmeasure*100,
            )

    return {key: result[key].mid.fmeasure*100 for key in score_keys}

def run_rouge(true_path, pred_path) :
    with open(true_path, "r", encoding='utf-8') as infile1:
        with open(pred_path, "r", encoding='utf-8') as infile2:
            pred_list = []
            real_list = []
            for line in infile1 :
                pred_list.append(line)
            for line2 in infile2 :
                real_list.append(line2)
            result = rouge(real_list, pred_list)
            print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_path", default='./models/base/validation_eval/korsmr_targets')
    parser.add_argument("--pred_path", default='./models/base/validation_eval/korsmr_746700_predictions')
    args = parser.parse_args()

    true_path = args.true_path
    pred_path = args.pred_path

    run_rouge(true_path, pred_path)