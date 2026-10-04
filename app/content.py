"""Static text of the website: navigation, figure captions, metric definitions, stage descriptions, related work.

Nothing here is a RESULT: captions say what a figure shows and why it matters, never what the numbers are. Every number
on the site is read from the project files at runtime (see site_data.py).
"""

NAV = [
    ("overview", "Overview", "/"),
    ("dataset", "Dataset", "/dataset"),
    ("preprocessing", "Preprocessing", "/preprocessing"),
    ("models", "Models & Results", "/models"),
    ("cross-domain", "Cross-domain", "/cross-domain"),
    ("postprocessing", "Postprocessing", "/postprocessing"),
    ("keyboard", "Keyboard", "/keyboard"),
    ("about", "About", "/about"),
]

PROBLEM = ("A keyboard has to guess the next word of a half-written sentence, and complete the word the user is typing, "
           "so that fewer keys have to be pressed. We train n-gram, RNN and LSTM language models on Shakespeare's complete "
           "works and ask how well they do it - and how badly a model trained on modern English does on the same text.")

PIPELINE = [
    ("Data", "Project Gutenberg #100, cut into individual works and split by work into train / val / test"),
    ("Preprocessing", "remove structure, directions and speaker tags; sentences; tokens; vocabulary"),
    ("Models", "KN n-grams, vanilla RNN, LSTM, ensemble - evaluated by one shared evaluator"),
    ("Postprocessing", "filter, prefix filter, case restoration, insertion, optional reranking"),
    ("Keyboard", "three suggestions, word completion, keystroke counter"),
]

# value-label -> (higher_is_better) used to highlight the best cell of a column
METRIC_HELP = {
    "ppl": ("Perplexity", False, "exp(mean negative log-probability) of the next token over ALL test tokens, including </s>, <unk>, <num> and punctuation. Lower is better."),
    "top1": ("Top-1", True, "Share of word targets for which the model's first suggestion is the actual next word. Targets that are <unk>, <num>, </s> or punctuation are not scored."),
    "top3": ("Top-3", True, "Share of word targets that are among the model's three suggestions (what the keyboard shows)."),
    "top5": ("Top-5", True, "Share of word targets among the model's five best suggestions."),
    "mrr": ("MRR", True, "Mean reciprocal rank of the actual word among all candidate words (1 = always first)."),
    "ksr": ("KSR", True, "Keystroke savings rate: 1 - keys pressed / keys without prediction, simulated letter by letter with three prefix-filtered suggestions; tapping a chip also inserts the space."),
    "archaic_top1": ("Archaic top-1", True, "Top-1 accuracy restricted to archaic-keyword targets (thou, thee, hath, doth, ... - tables/archaic_keywords.csv)."),
    "archaic_top5": ("Archaic top-5", True, "Top-5 accuracy restricted to archaic-keyword targets."),
    "latency_ms": ("Latency (ms, model call)", False, "Average wall-clock time of one next_word_probs call on the CPU (a stateless call: the RNN re-reads the prefix). NOT the end-to-end time in the app - see the numbers table."),
    "params": ("Parameters", False, "Neural models: trainable parameters (tied embeddings counted once). n-gram models: number of stored n-gram types."),
    "size_mb": ("Size on disk (MB)", False, "Size of the saved model file (pickle for n-grams, state dict for networks; the ensemble = LSTM + KN trigram)."),
}

# figure file -> (caption, data label). label: "test" | "val" | "train" | "all" (corpus / EDA) | "app"
FIGURES = {
    # --- dataset ---
    "words_per_work_by_category.png": ("Word count of every work, sorted and coloured by category.", "Shows how uneven the corpus is; the short poems are why they are kept in the training split and why per-category claims about Poetry and Romance rest on little data.", "all"),
    "category_words_and_works.png": ("Total words and number of works per category.", "Shows how much text each genre contributes, which limits what can be said about the small categories.", "all"),
    "zipf_rank_frequency.png": ("Rank-frequency plot of the vocabulary on log-log axes, with a fitted line.", "Shows that a few words account for most of the text and most words are rare: the long tail is the reason models need smoothing and a vocabulary cut.", "all"),
    "frequency_buckets.png": ("How many word types occur once, 2-5, 6-20 or more than 20 times, and how much of the running text each group covers.", "Shows the sparsity problem directly: many rare types, but they cover little text.", "all"),
    "line_length_histogram.png": ("Number of words per verse line.", "Shows why verse lines are short and why sentences, not lines, are the modelling unit.", "all"),
    "wordcloud_corpus_raw.png": ("Word cloud of the raw corpus, stop-words included.", "Shows what dominates the text before any cleaning: function words and the stage-direction and speaker words.", "all"),
    "wordcloud_corpus_no_stopwords.png": ("Word cloud of the raw corpus without stop-words.", "Shows the content words and, before preprocessing, how much speaker and stage-direction vocabulary (enter, exeunt, scene) is mixed in.", "all"),
    "wordclouds_by_category.png": ("Word clouds for comedies, tragedies and histories (stop-words removed).", "Shows how much vocabulary the genres share; character names are what mostly differ.", "all"),
    "distinctive_words_top30_excluding_names_bar.png": ("The 30 words most distinctive of Shakespeare compared with WikiText-2 (log-odds with an informative Dirichlet prior), after removing names and stage words and keeping words far more frequent than in modern English.", "These are the archaic words (thou, thy, hath, ...) used later to test whether a model predicts Shakespeare's own vocabulary.", "all"),
    "distinctive_words_top30_bar.png": ("The 30 highest-scoring words of the same comparison without filtering; grey bars are mostly speaker tags or stage words.", "Shows why the filtered list is needed: raw distinctiveness mixes archaic words with pronouns, names and stage directions.", "all"),
    "top20_ngrams_without_stopwords.png": ("Most frequent unigrams, bigrams and trigrams after removing stop-words.", "Shows that, before preprocessing, the most frequent content n-grams are play structure (exeunt, scene, act) rather than language.", "all"),
    # --- preprocessing ---
    "preprocessing_stage_counts.png": ("Tokens and vocabulary size after each preprocessing stage (blue: whitespace words, orange: tokens).", "Shows what each stage removes and where the vocabulary collapses (tokenisation, lowercasing, the frequency cut).", "all"),
    "preprocessing_wordclouds_before_after.png": ("Word clouds of the raw works and of the final preprocessed text.", "Shows that structure words disappear and the clouds become a picture of the language itself.", "all"),
    "preprocessing_sentence_length.png": ("Distribution of tokens per sentence after preprocessing.", "Shows the length range the models see; a few very long sentences (whole sonnets) matter for sequence length.", "all"),
    # --- n-grams / models (validation) ---
    "ngram_val_ppl_vs_n.png": ("Validation perplexity of Kneser-Ney and add-one models for n = 2 to 5.", "Shows how much context helps before sparsity takes over, and how much smoothing matters.", "val"),
    "ngram_val_ppl_vs_discount.png": ("Validation perplexity as a function of the discount d, per n-gram order; stars mark the best d.", "Shows the tuning used to choose the discount without touching the test set.", "val"),
    "ngram_val_topk_vs_n.png": ("Top-1, top-3 and top-5 validation accuracy versus n, for all word targets and for archaic targets.", "Shows whether better perplexity also means better suggestions.", "val"),
    "ngram_size_and_latency_vs_n.png": ("Number of stored n-grams and prediction latency versus n.", "Shows the cost side of higher orders.", "val"),
    "ngram_ksr_decomposition.png": ("Keystroke savings of a unigram and a KN trigram, with and without word completion.", "Separates how much of the saving comes from completing words by prefix and how much from the sentence context.", "val"),
    "rnn_training_curves.png": ("Training and validation perplexity per epoch for the vanilla RNN and the LSTM; the dashed line marks the epoch kept by early stopping.", "Shows convergence and the onset of overfitting.", "val"),
    "lstm_grid_val_curves.png": ("Validation perplexity per epoch for each of the eight LSTM configurations of the tuning grid.", "Shows the hyper-parameter search that was done on validation data only.", "val"),
    "rnn_ensemble_lambda.png": ("Validation perplexity of the ensemble P = lambda * P_LSTM + (1 - lambda) * P_KN3 for lambda from 0 to 1.", "Shows how the ensemble weight was chosen.", "val"),
    # --- test ---
    "rnn_test_comparison.png": ("Test perplexity, top-3 accuracy and keystroke savings of all models.", "The one-picture comparison of the final models on unseen plays.", "test"),
    "rnn_top1_by_position.png": ("Top-1 accuracy of the KN trigram and the LSTM by the position of the target word in the sentence.", "Shows where a recurrent model helps: it needs context, so the first word of a sentence is the same for both.", "test"),
    "rnn_top1_archaic_groups.png": ("Top-1 accuracy on thou / thee / thy / thine, on you / your, on archaic verb forms and on all other targets.", "Tests the author-aware part of the project: does the model know Shakespeare's own pronouns and verbs?", "test"),
    "robust_seeds.png": ("Test metrics for three training seeds (dots) and their mean (bar), with the KN trigram as reference.", "Shows that the differences between models are larger than the run-to-run variation.", "test"),
    "robust_bootstrap_ci.png": ("Relative difference between model pairs with 95 % paired-bootstrap confidence intervals over test sentences.", "Shows whether each improvement is statistically distinguishable from zero.", "test"),
    "robust_latency.png": ("Latency of the LSTM when the whole prefix is re-read (stateless) versus when only the newest token is fed (stateful), by prefix length.", "Shows that a real keyboard, which keeps the hidden state, needs a constant time per key.", "all"),
    # --- cross-domain ---
    "baseline_heatmaps.png": ("Top-3 accuracy and keystroke savings of KN-3 and LSTM for every combination of training and test domain.", "The core domain-transfer result: the off-diagonal cells show what happens when the training text is the wrong kind.", "test"),
    "baseline_archaic.png": ("Top-1 and top-5 accuracy on Shakespeare's archaic keywords for Shakespeare-trained and WikiText-trained models.", "Shows that a modern-English model cannot offer the words that make Shakespeare Shakespeare.", "test"),
    "baseline_wikitext_lstm_curve.png": ("Training curve of the WikiText LSTM (train in eval mode and WikiText validation).", "Shows that the baseline was trained to convergence with the same recipe.", "val"),
    # --- postprocessing ---
    "postproc_raw_special_rate.png": ("Which special tokens or punctuation marks are the raw top-1 prediction, as a share of test positions, for the LSTM and the KN trigram.", "Shows why the filtering stage is needed.", "test"),
    "postproc_wordclouds.png": ("Raw top-1 predictions, top-1 after filtering, and the actual target words over the test set.", "Shows the bias of arg-max prediction toward a few frequent words.", "test"),
    "rerank_comparison_val.png": ("Keystroke savings and top-1 accuracy with suggestions ordered by probability versus by expected keystrokes saved (validation).", "Tests the optional reranking stage; the decision to evaluate on test was based on this figure.", "val"),
    "rerank_ksr_by_word_length.png": ("Keystroke savings by target word length with and without reranking (validation).", "Shows where reranking helps (long words) and where it costs (short words).", "val"),
    # --- app ---
    "app_compare.png": ("Screenshot of the earlier compare-mode demo.", "Illustrates comparing two models on the same text.", "app"),
}

STAGE_TEXT = [
    ("1  Raw output", "The top-k tokens of the model's distribution over the whole vocabulary: may contain <s>, </s>, <unk>, <num> and punctuation; everything lowercase."),
    ("2  Filter", "Remove special tokens and punctuation; only real words stay. Probabilities are unchanged."),
    ("3  Prefix filter", "If the user is in the middle of a word, keep only words that start with the typed prefix and renormalise the probabilities over them."),
    ("4  Case restoration", "Capitalise at the start of a sentence; use the case map (most frequent mid-sentence casing in the training plays) for proper nouns and for I and O."),
    ("5  Insertion", "Replace the partial word with the chosen word plus a space; typing punctuation afterwards removes that space."),
    ("6  Reranking (optional)", "Order candidates by expected keystrokes saved, P(w) x (len(w) - typed letters), instead of by probability alone."),
]

# preprocessing stepper samples: (key, label, slug, anchor, lines before/after (line stages), sentences before/after)
SAMPLES = [
    ("hamlet_open", "Hamlet - opening of the play (header, Enter, speaker tags, 'Tis)", "hamlet_prince_of_denmark", "Nay, answer me", 13, 12, 4, 6),
    ("hamlet_aside", "Hamlet - an aside, a stage direction and Latin italics", "hamlet_prince_of_denmark", "A little more than kin", 4, 5, 3, 4),
    ("sonnet_2", "Sonnet 2 - sonnet number, single-quote quotation marks", "sonnets", "If thou couldst answer", 4, 9, 0, 0),
]

STAGE_DIRS = {1: "stage1_normalized", 2: "stage2_structure_removed", 3: "stage3_directions_removed", 4: "stage4_italics_stripped",
              5: "stage5_speaker_tags_removed", 6: "stage6_sentences", 7: "stage7_tokenized", 8: "stage8_lowercased"}

RELATED = [
    ("Kneser, R. and Ney, H. (1995). Improved backing-off for m-gram language modeling. Proc. ICASSP.", "interpolated Kneser-Ney smoothing"),
    ("Chen, S. F. and Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. Computer Speech & Language 13(4).", "discounting, continuation counts, interpolation"),
    ("Heafield, K. (2011). KenLM: faster and smaller language model queries. Proc. WMT.", "practical n-gram models; raw counts for <s>-initial n-grams"),
    ("Mikolov, T. et al. (2010). Recurrent neural network based language model. Proc. Interspeech.", "RNN language models"),
    ("Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8).", "the LSTM"),
    ("Press, O. and Wolf, L. (2017). Using the output embedding to improve language models. Proc. EACL; Inan, H., Khosravi, K. and Socher, R. (2017). Tying word vectors and word classifiers. Proc. ICLR.", "tied input/output embeddings"),
    ("Srivastava, N. et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR 15.", "dropout"),
    ("Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. Proc. ICLR.", "the optimiser"),
    ("Merity, S., Xiong, C., Bradbury, J. and Socher, R. (2017). Pointer sentinel mixture models. Proc. ICLR.", "the WikiText-2 corpus"),
    ("Monroe, B. L., Colaresi, M. P. and Quinn, K. M. (2008). Fightin' words: lexical feature selection and evaluation for identifying the content of political conflict. Political Analysis 16(4).", "log-odds with an informative Dirichlet prior"),
    ("Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.", "the paired bootstrap"),
    ("McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2).", "McNemar's test"),
    ("Zipf, G. K. (1949). Human Behavior and the Principle of Least Effort. Addison-Wesley.", "rank-frequency law"),
    ("Project Gutenberg. The Complete Works of William Shakespeare, eBook #100. https://www.gutenberg.org/ebooks/100", "the text"),
]

# files the pages need (relative to the project root). The startup check reports any that are missing.
REQUIRED_TABLES = [
    "corpus_overview", "vocabulary_stats", "metadata_dtypes", "category_summary", "split_summary", "split_works_per_category",
    "ngram_uniqueness", "frequency_buckets", "archaic_keywords", "preprocessing_stages", "preprocessing_residue",
    "preprocessing_oov_decomposition", "preprocessing_oov_mechanism", "preprocessing_oov", "preprocessing_rule_examples",
    "rnn_test_results", "robust_seeds_summary", "robust_seeds_per_run", "robust_bootstrap", "robust_mcnemar",
    "robust_latency", "rnn_lstm_grid", "rnn_vanilla_runs", "ngram_discount_tuning", "rnn_top1_by_position",
    "rnn_top1_archaic_groups", "rnn_examples", "rnn_samples", "baseline_cross_eval", "baseline_examples",
    "baseline_in_domain_vs_main", "baseline_data_summary", "postproc_stage_table", "postproc_raw_special_rate",
    "postproc_prediction_concentration", "postproc_case_accuracy", "postproc_case_accuracy_adjusted",
    "rerank_val_results", "rerank_test_results", "rerank_val_by_length", "rerank_examples", "app_latency",
    "ngram_val_results", "ngram_test_results",
]
REQUIRED_DATA = [
    "results/metrics.csv", "docs/key_numbers.md", "README.md", "data/split.csv", "data/metadata.csv",
    "data/processed/vocab.json", "data/processed/works_index.csv", "data/processed/wikitext/info.json",
    "app/demo_prefixes.txt",
] + [f"data/interim/{d}" for d in STAGE_DIRS.values()] + ["data/works"]
