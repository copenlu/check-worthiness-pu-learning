# Claim Check-Worthiness Detection as Positive Unlabelled Learning
Dustin Wright and Isabelle Augenstein

To appear in Findings of EMNLP 2020. Read the preprint: https://arxiv.org/abs/2003.02736

<p align="center">
  <img src="pu_learning_puc.png" alt="PUC">
</p>

A critical component of automatically combating misinformation is the detection of fact check-worthiness, ie determining if a piece of information should be checked for veracity. There are multiple isolated lines of research which address this core issue: check-worthiness detection from political speeches and debates, rumour detection on Twitter, and citation needed detection from Wikipedia. What is still lacking is a structured comparison of these variants of check-worthiness, as well as a unified approach to them. We find that check-worthiness detection is a very challenging task in any domain, because it both hinges upon detecting how factual a sentence is, and how likely a sentence is to be believed without verification. As such, annotators often only mark those instances they judge to be clear-cut check-worthy. Our best-performing method automatically corrects for this, using a variant of positive unlabelled learning, which learns when an instance annotated as not check-worthy should in fact have been annotated as being check-worthy. In applying this, we outperform the state of the art in two of the three domains studied for check-worthiness detection in English.

# Citing

```bib
@inproceedings{wright2020claim,
    title={{Claim Check-Worthiness Detection as Positive Unlabelled Learning}},
    author={Dustin Wright and Isabelle Augenstein},
    booktitle = {Findings of EMNLP},
    publisher = {Association for Computational Linguistics},
    year = 2020
}
```

# Recreating Results

To recreate our results, first navigate to each individual 'data' directory and follow the instructions to download the data used in the paper.

Create a new conda environment:

```bash
$ conda create --name check-worthiness-pu python=3.7
$ conda activate check-worthiness-pu
$ pip install -r requirements.txt
```

Note that this project uses wandb; if you do not use wandb, set the following flag to store runs only locally:

```bash
export WANDB_MODE=dryrun
```

## Citation Needed Results
Baseline Wikipedia:

```bash
python train_wikipedia.py \
    --pos_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_all_citations_sample.txt" \
    --neg_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_no_citations_sample.txt" \
    --train_pct 0.8 \
    --n_gpu 1 \
    --log_interval 1 \
    --test_files "data/wikipedia/english_citation_data/rnd - random articles/all_citations_sample.txt" "data/wikipedia/english_citation_data/rnd - random articles/no_citations_sample.txt" \
    --test_files "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_cn_citations_sample.txt" "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_no_citations_sample.txt" \
    --seed 1000 \
    --model_dir models/wikipedia
```

Wikipedia with PU Learning:

```bash
python train_wikipedia_pu_learning.py \
  --pos_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_all_citations_sample.txt" \
  --neg_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_no_citations_sample.txt" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --test_files "data/wikipedia/english_citation_data/rnd - random articles/all_citations_sample.txt" "data/wikipedia/english_citation_data/rnd - random articles/no_citations_sample.txt" \
  --test_files "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_cn_citations_sample.txt" "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_no_citations_sample.txt" \
  --pretrained_model "models/wikipedia/model.pth" \
  --seed 1000 \
  --model_dir models/pu-wikipedia
  --indices_dir models/wikipedia
```

Wikipedia with positive unlabelled conversion:

```bash
python train_wikipedia_puc.py \
  --pos_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_all_citations_sample.txt" \
  --neg_file "data/wikipedia/english_citation_data/fa - featured articles/en_wiki_subset_statements_no_citations_sample.txt" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --test_files "data/wikipedia/english_citation_data/rnd - random articles/all_citations_sample.txt" "data/wikipedia/english_citation_data/rnd - random articles/no_citations_sample.txt" \
  --test_files "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_cn_citations_sample.txt" "data/wikipedia/english_citation_data/lqn - citation needed articles/statements_no_citations_sample.txt" \
  --pretrained_model "models/wikipedia/model.pth" \
  --seed 1000 \
  --model_dir models/puc-wikipedia
  --indices_dir models/wikipedia
```

Baseline Pheme:

```bash
python train_pheme.py \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/pheme
```

Pheme + PU or PUC:

```bash
python {train_pheme_pu_learning.py|train_pheme_puc.py} \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/{pheme-pu-solo|pheme-puc-solo}
  --pretrained_pheme_model models/pheme
  --indices_dir models/pheme
```

Pheme + Wiki:

```bash
python train_pheme.py \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/{pheme-wiki|pheme-pu|pheme-puc} \
  --pretrained_model "models/{wikipedia|pu-wikipedia|puc-wikipedia}/model.pth"
  --indices_dir models/pheme
```

Pheme + PU or PUC + Wiki:

```bash
python {train_pheme_pu_learning.py|train_pheme_puc.py} \
  --pheme_dir "data/pheme/" \
  --train_pct 0.8 \
  --n_gpu 1 \
  --log_interval 1 \
  --exclude_splits ebola-essien gurlitt prince-toronto putinmissing \
  --seed 1000 \
  --model_dir models/{pheme-pu-wiki|pheme-puc-wiki}
  --pretrained_pheme_model models/pheme/model
  --pretrained_model "models/{pu-wikipedia|puc-wikipedia}/model.pth"
  --indices_dir models/pheme
```

To run the Clef experiments, first download the clef 2018 repo which has the official scoring script: https://github.com/clef2018-factchecking/clef2018-factchecking

Replace "clef_scorer.py" with the location of the file "scorer/task1.py" from that repo. Make sure to add the repo to your python path as well.

Baseline Clef:

```bash
python train_clef.py \
    --split_dir "data/clef" \
    --train_pct 0.9 \
    --n_gpu 1 \
    --log_interval 1 \
    --clef_test_script "clef_scorer.py" \
    --seed 1000 \
    --model_dir models/clef \
```

Clef + PU or PUC:

```bash
python {train_clef_pu_learning.py|train_clef_puc.py} \
    --split_dir "data/clef" \
    --train_pct 0.9 \
    --n_gpu 1 \
    --log_interval 1 \
    --clef_test_script "clef_scorer.py" \
    --seed 1000 \
    --pretrained_clef_model models/clef
    --model_dir models/{clef-pu-solo|clef-puc-solo} \
    --indices_dir models/clef
```

Clef + Wiki:

```bash
python train_clef.py \
    --split_dir "data/clef" \
    --train_pct 0.9 \
    --n_gpu 1 \
    --log_interval 1 \
    --seed 1000 \
    --pretrained_model "models/{wikipedia|pu-wikipedia|puc-wikipedia}/model.pth"
    --model_dir models/{clef-wiki|clef-pu|clef-puc}
    --indices_dir models/clef
```

Clef + PU or PUC + Wiki:

```bash
python {train_clef_pu_learning.py|train_clef_puc.py} \
    --split_dir "data/clef" \
    --train_pct 0.9 \
    --n_gpu 1 \
    --log_interval 1 \
    --clef_test_script "clef_scorer.py" \
    --seed 1000 \
    --pretrained_clef_model models/clef
    --pretrained_model "models/{pu-wikipedia|puc-wikipedia}/model.pth"
    --model_dir models/{clef-pu-solo|clef-puc-solo} \
    --indices_dir models/clef
```
