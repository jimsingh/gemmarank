# T5 Gemma Ranking Research Engineering

Much of the recent excitement in language models today has focused on generative / causal models. So much so that I recently trained my own [GPT-style model](github.com/jimsingh/llm_e2e) for fun. Google recently released [T5Gemma](https://developers.googleblog.com/en/t5gemma/) and I'm excited to see some 'attention' being given to encoder-decoder / seq-to-seq architectures.

In the past, we've seen that encoder heavy architectures such as BERT or T5 are more parameter efficient rankers than other model architectures. This is perhaps because these models use bidirectional attention and are trained with span corruption. I've been impressed with the Gemma model releases and this project explores how T5Gemma could be finetuned for neural ranking.

## Background

The last time I looked at T5 it was the [RankT5 paper](https://arxiv.org/abs/2210.10634), which fine-tuned t5x to take the first token's hidden state from the encoder and passes it through a linear layer to compute relevance scores.

One gap that I noticed in the paper was that the loss function comparison was limited to pairwise and listwise. The quadratic nature of pairwise loss limits its scalability to longer result sets and the paper found that evaluating more results with a listwise loss function was superior to using pairwise loss on a smaller resultset.

My hypothesis is that for real world use cases we need to consider both top-5 result pairwise quality *and* listwise for the remainder. This could capture both aspects of ranking: precision at the top for question / answer and quality browsing lists for users that want to go deeper by exploring lower positions.

One criticism of this might be that users typically only click on the top results. My retort is that this is because we've trained them to by providing low quality results after position 5 (give or take). Users clearly scroll far beyond position 5 when they are in an exploratory mood (instagram, tiktok).

## Project Goals

- Finetune T5 to get close to the original paper's performance
- Replicate the same with T5Gemma
- Train with a top-5 pairwise + listwise hybrid loss 
- Evaluate MS MARCO passage ranking using standard IR metrics (MRR@10, NDCG@5)
- Evaluate performance on a combination of question answer and browsy queries 

## Methodology

- build a straightforward framework to eval ranking tasks (trec_eval, ranx, pyserini)
- use BM25 based retrieval and a (very) low quality TF-IDF ranker
- substitute neural ranking starting with off the shelf models
- finetune T5X and T5Gemma for ranking
- evaluate using standard IR metrics
- examine results qualitatively
