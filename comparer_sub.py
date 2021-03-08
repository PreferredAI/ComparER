import os
import glob
import cornac
import numpy as np
from cornac.utils import cache
from cornac.models import ComparERSub
from cornac.metrics import AUC, Recall, NDCG
from cornac.datasets import amazon_toy
from cornac.experiment import Experiment
from cornac.data.reader import Reader
from cornac.eval_methods import StratifiedSplit
from cornac.data.sentiment import SentimentModality


rating = amazon_toy.load_feedback(fmt="UIRT", reader=Reader(min_user_freq=10))
sentiment_data = amazon_toy.load_sentiment()
md = SentimentModality(data=sentiment_data)

eval_method = StratifiedSplit(
    rating,
    group_by="user",
    chrono=True,
    sentiment=md,
    test_size=0.2,
    val_size=0.16,
    exclude_unknowns=True,
    verbose=True,
)

pretrained_mter_model_path = "dist/toy/result/MTER"
params = {}
if os.path.exists(pretrained_mter_model_path):
    pretrained_path = max(
        glob.glob(os.path.join(pretrained_mter_model_path, "*.pkl")),
        key=os.path.getctime,
    )  # get latest saved model path
    pretrained_model = np.load(pretrained_path, allow_pickle=True)
    params = {
        "G1": pretrained_model.G1,
        "G2": pretrained_model.G2,
        "G3": pretrained_model.G3,
        "U": pretrained_model.U,
        "I": pretrained_model.I,
        "A": pretrained_model.A,
        "O": pretrained_model.O,
    }

model = ComparERSub(
    name="ComparERSub",
    n_user_factors=8,
    n_item_factors=8,
    n_aspect_factors=8,
    n_opinion_factors=8,
    n_pair_samples=1000,
    n_bpr_samples=1000,
    n_element_samples=50,
    lambda_reg=0.1,
    lambda_bpr=10,
    lambda_d=10,
    max_iter=10000,
    lr=0.5,
    min_common_freq=1,
    min_user_freq=2,
    min_pair_freq=1,
    trainable=True,
    verbose=True,
    init_params=params,
)

n_items = eval_method.train_set.num_items

k_1 = int(n_items / 100)
k_5 = int(n_items * 5 / 100)
k_10 = int(n_items * 10 / 100)

Experiment(
    eval_method,
    models=[model],
    metrics=[
        AUC(),
        Recall(k=k_1),
        Recall(k=k_5),
        Recall(k=k_10),
        NDCG(k=k_1),
        NDCG(k=k_5),
        NDCG(k=k_10),
    ],
    show_validation=True,
    save_dir="dist/toy/result",
    verbose=True,
).run()
