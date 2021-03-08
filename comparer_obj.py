import os
import glob
import cornac
import numpy as np
from cornac.utils import cache
from cornac.models import ComparERObj
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


pretrained_mter_model_path = "dist/toy/result/EFM"
params = {}
if os.path.exists(pretrained_mter_model_path):
    pretrained_path = max(
        glob.glob(os.path.join(pretrained_mter_model_path, "*.pkl")),
        key=os.path.getctime,
    )  # get latest saved model path
    pretrained_model = np.load(pretrained_path, allow_pickle=True)
    params = {
        "U1": pretrained_model.U1,
        "U2": pretrained_model.U2,
        "H1": pretrained_model.H1,
        "H2": pretrained_model.H2,
        "V": pretrained_model.V,
    }

model = ComparERObj(
    name="ComparERObj",
    num_explicit_factors=128,
    num_latent_factors=128,
    num_most_cared_aspects=20,
    rating_scale=5.0,
    alpha=0.7,
    lambda_x=1,
    lambda_y=1,
    lambda_u=0.01,
    lambda_h=0.01,
    lambda_v=0.01,
    lambda_d=0.1,
    min_user_freq=2,
    max_iter=1000,
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
