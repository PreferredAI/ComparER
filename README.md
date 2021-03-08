# Explainable Recommendation with Comparative Constraints on Product Aspects

This is the code for the paper:

**[Explainable Recommendation with Comparative Constraints on Product Aspects](https://lthoang.com/assets/publications/wsdm21.pdf)**
<br>
[Trung-Hoang Le](http://lthoang.com/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [WSDM 2021](https://www.wsdm-conference.org/2021/)


If you find the code and data useful in your research, please cite:

```
@inproceedings{10.1145/3437963.3441754,
  title     = {Explainable Recommendation with Comparative Constraints on Product Aspects},
  author    = {Le, Trung-Hoang and Lauw, Hady W.},
  year      = {2021},
  isbn      = {9781450382977},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3437963.3441754},
  doi       = {10.1145/3437963.3441754},
  booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages     = {967–975},
  numpages  = {9},
  keywords  = {explainable recommendation, comparative constraints},
  location  = {Virtual Event, Israel},
  series    = {WSDM '21}
}
```

## How to run

```bash
pip install -r requirements.txt
```

There are two variants of ComparER model: subjective and objective. 

### Run ComparER on Subjective Aspect-Level Quality

Run MTER model:
```bash
python mter.py
```

MTER is the base model of ComparER with subjective aspect-level quality. After finish training MTER, we can continue train ComparERSub by the command:
```bash
python comparer_sub.py
```

### Run ComparER on Objective Aspect-Level Quality

Run EFM model:
```bash
python efm.py
```

EFM is the base model of ComparER with objective aspect-level quality. After finish training EFM, we can continue train ComparERObj by the command:
```bash
python comparer_obj.py
```

## Contact
Questions and discussion are welcome: [lthoang.com](http://lthoang.com)
