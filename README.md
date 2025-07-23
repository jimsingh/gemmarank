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
- T5Gemma is a stretch goal once I understand how it's trained
- Train with a top-5 pairwise + listwise hybrid loss 
- Evaluate MS MARCO passage ranking using standard IR metrics (MRR@10, NDCG@5)
- Evaluate performance on a combination of question answer and browsy queries 

## Progress

### representation extraction 

I evaluated three approaches for caputring the T5 encoder's representational embeddings from the hidden layer.

1) first token
2) last token
3) pooling (averaging)
3b) pooling with attention

I did this by comparing the cosine similarity of postiive and negative examples using each of these three approaches. pooling with attention (masking out the tokens that were not attended to and/or pading) produced the lowest cosine similarity score, which told me that this was the strongest starting signal.

### training issues and fixes

In my first training run

1. the model quickly overfit, showing growing diference between validation loss and training loss
2. the model continued to drive down loss by driving up separation, likely pushing easy examples further apart
3. using the same learning rate and decay of the entire model didn't intuitively feel like the right strategy. 
4. observed a sawtooth loss pattern after the model completed an epoch (showing overfit)

So, I did the following
1. added tanh to the loss to cap the reward for separating easy examples (quick fix - long term I'd use margin or focal loss)
2. added drop out 0.1
3. used different LRs for embedings, encoder, and dense layers
4. switched to BCE (also a quick fix - should be margin-based for ranking)
5. decreased AdamW beta2 to 0.97 (down from the default of 0.999) to reduce the second moment estimation window for more responsive parameter updates
6. shuffled far more of the dataset because I saw a saw tooth loss pattern
7. logged more data during training (separation, parameter drift over time by layer, lr decay by layer -- at this point I was just trying to see what I could do with wandb)

### training results

The charts compare two configurations:
- **Red line**: T5-Large with differential learning rates only
- **Yellow line**: T5-Large with differential learning rates + regularization (dropout)

![Separation](assets/separation.png)
![Training Loss](assets/train_loss.png)  
![Validation Loss](assets/val_loss.png)

Looking at the validation loss chart, you can see the non-regularized model (red) starts overfitting with validation loss climbing from ~0.47 to ~0.57. The regularized model (yellow) keeps validation loss stable around 0.44-0.45. Both models get similar separation (~2.5-3.0 range) but the regularized version is much more stable.

The differential learning rates I used:
```python
embedding_lr = 5e-6    # conservative for delicate embeddings
encoder_lr = 5e-5      # moderate for pretrained encoder  
dense_lr = 2e-4        # aggressive for untrained dense layer
```

### what's next

- finish T5-Large baseline evaluation on MS MARCO
- implement the hybrid pairwise+listwise loss function
- evaluation on question-answer vs exploratory queries
- T5Gemma once I understand HOW it was trained better - it seems to have transferred weights from GemmaV2

## Methodology

- build a straightforward framework to eval ranking tasks (trec_eval, ranx, pyserini)
- use BM25 based retrieval and a (very) low quality TF-IDF ranker
- substitute TF-iDF with neural ranking starting with off the shelf models
- finetune T5X and T5Gemma for ranking
- evaluate using standard IR metrics
- examine results qualitatively
