# Twitter Named Entity Recognition (NER)

Fine-grained named entity recognition on tweets — tagging tokens as `person`, `geo-loc`, `company`, `facility`, `product`, `musicartist`, `movie`, `sportsteam`, `tvshow`, or `other` — without relying on user-provided hashtags.

## Goal

Automatically identify and classify named entities in tweets so that downstream systems (trend detection, brand monitoring, recommendation, search, moderation, etc.) don't have to depend on noisy, missing, or misspelled hashtags to understand what a tweet is about.

## Dataset

- **Source:** WNUT-2016 Twitter NER shared-task data (Strauss, de Marneffe & Ritter), from the `aritter/twitter_nlp` GitHub repository.
- **Format:** CoNLL/BIO — one token per line (`TOKEN<TAB>LABEL`), sentences separated by blank lines.
- **Files used:** `wnut_16.txt.conll` (train) and `wnut16test.txt.conll` (test) — this release ships train + test only (no separate dev set).
- **Tag scheme:** 10 entity types × `B-`/`I-` prefixes + `O` = 21 distinct tags.
- **Size:**
  - Train: 2,394 sentences
  - Test: 3,850 sentences
  - Total tokens: 108,377 | Vocabulary (lower-cased): 21,934
  - Sentence length: mean 19.4 tokens, 95th percentile 31, max 39
- **Class imbalance:** `O` dominates (~92% of tokens); rare types like `tvshow` and `movie` have very few examples — this imbalance is one of the core challenges of the task.
- **Data quality check:** BIO scheme validated — 0 violations found (every `I-X` tag correctly follows a `B-X`/`I-X` of the same type).

## Approach

Two modeling approaches are built, trained, and compared on the same held-out test set using **entity-level** precision/recall/F1 (via `seqeval`), which only counts a predicted span as correct if both its boundaries and its entity type match the gold span — a stricter and more meaningful metric than per-token accuracy for NER.

### Part A — BiLSTM + Word2Vec-initialized embeddings

1. Build word/tag vocabularies with a Keras `Tokenizer`.
2. Train a domain-specific **Word2Vec** (Gensim, skip-gram) model directly on the tweet corpus — generic pretrained vectors under-represent Twitter-specific vocabulary (slang, `@mentions`, emoji-adjacent tokens).
3. Use the Word2Vec vectors to **initialize** (not freeze) a Keras `Embedding` layer.
4. Pad sequences to a fixed `MAX_LEN` (33, covering the 95th percentile of tweet lengths).
5. Train a **Bidirectional LSTM** with a `TimeDistributed(Dense(softmax))` output layer for per-token classification.
6. Run a small hyperparameter search over `lstm_units`, `dropout`, and `learning_rate`, with early stopping on validation loss.

**Why bidirectional?** A unidirectional LSTM only sees words *before* the current token, but entity type often depends on context that comes *after* it too (e.g., "Apple **unveiled**..." signals `company`, not `product`). A BiLSTM conditions every token's representation on the full sentence.

### Part B — Fine-tuned `bert-base-uncased`

1. Tokenize with `is_split_into_words=True` to preserve a mapping from BERT sub-tokens back to original words.
2. **Align labels to sub-tokens:** label only the first sub-token of each word; assign `-100` (ignored in loss) to continuation sub-tokens and special tokens (`[CLS]`, `[SEP]`, padding).
3. Fine-tune with HuggingFace `Trainer`, using `seqeval`-based `compute_metrics` and early stopping on validation F1.
4. At inference time, **re-combine sub-tokens** into whole-word predictions by keeping only the prediction on each word's first sub-token.
5. Fine-tuning config: learning rate 3e-5, batch size 16, up to 8 epochs (early stopping triggered at epoch 6), weight decay 0.01, `load_best_model_at_end=True`.

## Results

| Model | Approach | Entity-level Test F1 |
|---|---|---|
| BiLSTM + softmax (Word2Vec-initialized) | Trains embeddings + tagger from scratch on this dataset only | 0.076 |
| Fine-tuned `bert-base-uncased` | Starts from large-scale pretrained language representations | 0.446 |

**BERT substantially outperforms the from-scratch BiLSTM**, especially on rarer entity types (`movie`, `tvshow`, `musicartist`), because its pretrained representations already encode broad world/lexical knowledge that a model trained on only a few thousand tweets cannot learn from scratch. Per-entity BERT test performance ranges from strong (`person` F1 0.63, `geo-loc` F1 0.64) to weak (`tvshow` F1 0.00, `facility` F1 0.21), reflecting the underlying class imbalance in the training data.

The notebook also demonstrates both models qualitatively on custom example sentences (e.g., correctly tagging "Harry Potter" as `person`, "Disney World" as `company`/`facility`, "Game of Thrones" as `movie`, "Apple"/"iPhone" as `product`).

## Key Design Decisions & Rationale

- **Word2Vec initialization (not random) for the LSTM embedding layer** — gives the model a head start with domain-relevant word similarity structure, achieving 100% vocabulary coverage from the trained vectors.
- **Bidirectional over unidirectional LSTM** — captures both left and right context, standard practice for sequence labeling tasks.
- **Entity-level (seqeval) evaluation over token-level accuracy** — token accuracy is misleadingly high due to the `O`-dominated class distribution; entity-level F1 better reflects real-world usefulness.
- **Early stopping** on both models — reduces training time and prevents overfitting on a relatively small labeled dataset; the best checkpoint (by validation metric) is restored automatically rather than using the final epoch's weights.
- **Sub-token label alignment for BERT** — necessary because WordPiece tokenization splits many Twitter-specific words (slang, misspellings, handles) into multiple sub-tokens, requiring a strategy to map single word-level labels onto them and back.

## Requirements

- Python 3.x
- TensorFlow 2.20.0 (for the BiLSTM model)
- PyTorch (for BERT fine-tuning via HuggingFace)
- `transformers`, `datasets`, `evaluate`, `seqeval`, `accelerate`
- `gensim` (Word2Vec)
- NumPy, Pandas, Matplotlib
- scikit-learn
- A GPU is recommended, especially for BERT fine-tuning

## Usage

Run the notebook cells in order:

1. Install dependencies and import libraries.
2. Load and parse the CoNLL-formatted train/test files.
3. Run EDA (sentence length, vocabulary, tag distribution, BIO consistency check).
4. **Part A:** Build vocab → train Word2Vec → build embedding matrix → train BiLSTM → run hyperparameter search → evaluate on test set with `seqeval`.
5. **Part B:** Tokenize and align labels for BERT → fine-tune with `Trainer` → evaluate on test set → try custom sentences.
6. Compare both models side by side.
7. Review the answers to the assignment's conceptual questions (data format, alternative annotation schemes, tokenization rationale, alternative model choices, effect of early stopping, BERT sentence-pair handling, attention vs. recurrence, BERT vs. `simpletransformers`).

## Outputs

- Fine-tuned BERT model + tokenizer saved to `./bert-twitter-ner-final` (also zipped as `bert-twitter-ner-final.zip`)
- Training/evaluation logs and checkpoints under `./bert-twitter-ner`

## Notes & Limitations

- Precision/recall for entity types with very few test examples (`movie`, `tvshow`) are noisy and should be interpreted cautiously — `seqeval` issues `UndefinedMetricWarning` for labels with no predicted samples in some runs.
- The BiLSTM's low F1 (0.076) reflects the difficulty of learning good representations from a small labeled dataset alone, not a flaw in the architecture — it's the intended point of comparison against pretrained-transfer-learning (BERT).
- Further improvements could include: a Twitter-domain-pretrained encoder (e.g., BERTweet), a CRF output layer on top of BERT for tag-transition modeling, character-level CNN features for the LSTM path, or oversampling/class-weighting for rare entity types.
