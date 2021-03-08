import os
import cornac
from cornac.utils import cache
from cornac.models import EFM
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

model = EFM(
    name="EFM",
    num_explicit_factors=128,
    num_latent_factors=128,
    num_most_cared_aspects=100,
    rating_scale=5.0,
    alpha=0.9,
    lambda_x=1,
    lambda_y=1,
    lambda_u=0.01,
    lambda_h=0.01,
    lambda_v=0.01,
    max_iter=10000,
    trainable=True,
    verbose=True,
)

n_items = eval_method.train_set.num_items

k_1 = int(n_items / 100)
k_5 = int(n_items * 5 / 100)
k_10 = int(n_items * 10 / 100)

cornac.Experiment(
    eval_method=eval_method,
    models=[model],
    metrics=[
        cornac.metrics.AUC(),
        cornac.metrics.Recall(k=k_1),
        cornac.metrics.Recall(k=k_5),
        cornac.metrics.Recall(k=k_10),
        cornac.metrics.NDCG(k=k_1),
        cornac.metrics.NDCG(k=k_5),
        cornac.metrics.NDCG(k=k_10),
    ],
    show_validation=True,
    save_dir="dist/toy/result",
    verbose=True,
).run()
