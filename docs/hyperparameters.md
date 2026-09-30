# Settings reference

Every setting PRISMT understands, with its default and what it does. The same text appears
as tooltips in the app (Model & training tab > Show all settings) and in
`prismt.defaultConfig`'s help. This page is generated from
`src/prismt/resources/config_schema.json` by `tools/make_settings_doc.py`; do not edit it by hand.

## How settings are combined

A run's settings are built in three layers, each overriding the one before:

1. the defaults below;
2. the chosen **preset** (Quick test, Standard or Paper), which sets model size and training
   budget;
3. what you set yourself, in the app or in the `cfg` struct of a script.

The resolved settings of every run are saved in its `config.json`, so a run can always be
repeated exactly (Results tab > Export as MATLAB script).

## What to change first

Most analyses only need the **Task** tab (what to predict, which trials, how to test) and a
preset. When a result looks wrong, the usual fixes are, in order:

- **The model does no better than the reference models.** Train longer (`train.epochs`,
  `train.min_epochs`), or try the Standard preset. Check the learning curves: a validation
  loss that is still falling at the last epoch means the budget was too small.
- **Validation loss rises early while training loss keeps falling (overfitting).** Use a
  smaller model (`model.d_model`, `model.n_layers`), more dropout (`model.dropout`), or more
  data (fewer trial filters, all sessions).
- **Training is too slow or runs out of memory.** Use fewer tokens: a shorter
  `preprocess.time_window_s`, wider `preprocess.bin_width_s`, or `model.time_patch` > 1
  (or `all`, one token per channel). The Check reports the number of tokens per trial; above
  about 1,100 a laptop becomes slow.
- **Loss becomes NaN.** Lower `train.lr` (for example 3e-4).
- **You want the best settings found for you.** Use automatic tuning (`hpo`), ideally on a
  cluster; it compares settings on validation trials only and tests the winner once.

## Presets

| Preset | What it is for |
|---|---|
| Quick test (`quick`) | Small model, a few minutes on a laptop. Checks that everything works. |
| Standard (`standard`) | Larger model for real analyses; tens of minutes on a laptop GPU, faster on a cluster. |
| Paper (`paper`) | The PRISMt paper's configuration (model width 512, 90% masking, batch 128, learning rate 5e-4, up to 200 epochs). Needs a GPU. |

Settings that the presets change:

| Setting | Quick test | Standard | Paper |
|---|---|---|---|
| `model.d_model` | 64 | 128 | 512 |
| `model.n_layers` | 2 | 4 | 4 |
| `model.n_heads` | 4 | 4 | 8 |
| `train.batch_size` | 32 | 64 | 128 |
| `train.lr` | 0.001 | 0.0005 | 0.0005 |
| `train.epochs` | 80 | 150 | 200 |
| `train.patience` | 15 | 20 | 20 |
| `train.min_epochs` | 20 | 30 | 30 |
| `model.position_embedding` | — | — | per_token |
| `train.weight_decay` | — | — | 0.01 |
| `train.grad_clip` | — | — | 1.0 |
| `mae.mask.ratio` | — | — | 0.9 |

## Task (`task`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `task` | classify | `classify` / `mae` | **What to learn.** classify: learn to tell trial conditions apart (for example early vs late learning, or CS+ vs CS-) from the recorded signals. mae (masked autoencoder): learn the structure of the signals by hiding part of each trial and predicting it; needs no labels. |

## Data (`dataset`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `dataset.path` | none | file or folder | **Dataset file.** A PRISMT dataset (.mat) written by prismt.writeDataset or the Data tab. |

## Which trials (`selection`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `selection.filters` | (empty) | filters | **Trial filters.** Keep only trials that match every filter, for example stim is CS+, or response is hit or CR. |
| `selection.channels` (advanced) | none | list | **Channels.** Names of the channels to use. Empty: all channels. |
| `selection.modalities` | none | list | **Modalities.** Names of the modalities (signals) to use, for example calcium and ach. Empty: all. |
| `selection.max_trials_per_session` (advanced) | none | whole number, at least 1 | **Trials per session (at most).** Use at most this many randomly chosen trials from each session, to make a test run faster. Empty: all trials. |
| `selection.min_valid_fraction` (advanced) | 0.5 | number, 0 to 1 | **Minimum data present per trial.** Skip a trial if less than this fraction of its values are present (not missing). |

## Classes (`labels`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `labels.column` | none | text | **Label column.** The trial column whose values the model learns to tell apart, for example phase or stim. |
| `labels.classes` | (empty) | classes | **Classes.** Which values to compare. Values given the same class name are merged (for example hit and CR as 'correct'). Empty: every value is its own class. |
| `labels.class_weighting` (advanced) | balanced | `balanced` / `none` | **Class weighting.** balanced: rare classes count as much as common ones, so the model cannot score well by always guessing the most common class. |

## Time window and scaling (`preprocess`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `preprocess.time_window_s` | none | start, end (s) | **Time window (s).** Use only samples between these two times, in seconds relative to the event (for example 0 to 3). Empty: the whole trial. |
| `preprocess.bin_width_s` | none | number, at least 0 | **Time bin width (s).** Average consecutive samples into bins of this width. Fewer time bins make training much faster: the model's cost grows with the square of (time bins x channels). Empty: keep every sample. |
| `preprocess.baseline_subtract` (advanced) | off | on / off | **Subtract pre-event baseline.** Subtract each trial's average before time 0, channel by channel. |
| `preprocess.normalize` (advanced) | zscore_train | `zscore_train` / `none` | **Scaling.** zscore_train: scale each channel to mean 0 and standard deviation 1, computed on training trials only (never on test trials). none: use the values as they are. |
| `preprocess.clip_sd` (advanced) | 10 | number, at least 0 | **Clip at (standard deviations).** After scaling, limit values to plus or minus this many standard deviations so rare artifacts cannot dominate training. Empty: no limit. |

## Testing (`split`)

How trials are divided into training, validation (choosing when to stop) and test (the reported score). See docs/results-and-pitfalls.md for why testing on new animals matters.

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `split.test_on` | auto | `auto` / `subject` / `session` / `trial` | **Test on.** What the final score must generalize to. subject: animals never used for training (the strongest claim). session: new sessions of known animals. trial: held-out trials of known sessions, only valid when the label changes within sessions. auto: the strongest level your data allow. |
| `split.validate_on` (advanced) | auto | `auto` / `subject` / `session` / `trial` | **Validate on.** Validation trials decide when to stop training and which epoch to keep. They are held out at this level from the training data. auto: the same level as testing when there are enough groups. |
| `split.folds` | auto | number or `auto` | **Evaluation.** auto: one train/validation/test split when there are 10 or more groups, otherwise cross-validation so that every animal is tested once. A number k: k-fold cross-validation. loo: leave one group out (each animal is the test set once). |
| `split.test_fraction` (advanced) | 0.2 | number, 0 to 0.5 | **Test fraction.** Fraction of groups used for testing when there is a single split. |
| `split.val_fraction` (advanced) | 0.15 | number, 0 to 0.5 | **Validation fraction.** Fraction of the training groups held out for validation. |
| `split.seed` (advanced) | 0 | whole number, at least 0 | **Split seed.** Changes which animals end up in training, validation and test. Results should not depend on it much. |
| `split.allow_leaky` (advanced) | off | on / off | **Allow leaky split.** Allow the same session or animal in training and testing even though the label is constant within it. Scores will be inflated; for exploration only. |

## Model (`model`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `model.d_model` | 64 (preset) | `16` / `32` / `64` / `128` / `256` / `512` | **Model width.** Size of the internal representation of each token. Larger models can learn more, but need more data and time. |
| `model.n_layers` | 2 (preset) | whole number, 1 to 12 | **Layers.** Number of transformer layers. More layers can combine information in more complex ways. |
| `model.n_heads` (advanced) | 4 (preset) | whole number, 1 to 16 | **Attention heads.** Number of parallel attention patterns in each layer. Must divide the model width. |
| `model.ff_mult` (advanced) | 4 | whole number, 1 to 8 | **Feed-forward size (x width).** Width of each layer's feed-forward network, as a multiple of the model width. |
| `model.dropout` (advanced) | 0.1 | number, 0 to 0.6 | **Dropout.** Randomly drops parts of the network during training to reduce overfitting. |
| `model.attention` (advanced) | block_causal | `block_causal` / `full` | **Attention.** block_causal (as in the PRISMt paper): each time bin can use the present and the past, never the future. full: every token can use every other token. |
| `model.time_patch` (advanced) | 1 | number or `all` | **Time bins per token.** 1: one token per value (the paper's model). A larger number groups that many consecutive time bins of a channel into one token: faster, but coarser in time. all: one token per channel. |
| `model.position_embedding` (advanced) | channel_time (preset) | `channel_time` / `per_token` | **Position encoding.** channel_time: the model learns one code per channel and one per time bin, and adds them. per_token: one code per (channel, time) token, as in the PRISMt paper; needs more data. |
| `model.init_from` | none | file or folder | **Start from autoencoder.** Start the classifier from a finished masked-autoencoder run on the same dataset (fine-tuning). Give the run folder. |
| `model.freeze_encoder` (advanced) | off | on / off | **Freeze autoencoder.** When starting from an autoencoder, train only the classification head and keep the rest fixed. |

## Masked autoencoder (`mae`)

Only used by masked-autoencoder runs.

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `mae.mask.strategy` | random | `random` / `channel` / `forecast` / `modality` | **What to hide.** random: hide random tokens. channel: hide whole channels (which channels are predictable from the others?). forecast: hide the end of each trial (predict the future). modality: hide one modality (predict, for example, ACh from calcium). |
| `mae.mask.ratio` | 0.9 (preset) | number, 0.05 to 0.95 | **Fraction hidden.** For random: the fraction of tokens hidden (the PRISMt paper used 0.9). For channel: the fraction of channels hidden. |
| `mae.mask.context_fraction` (advanced) | 0.5 | number, 0.05 to 0.95 | **Forecast: fraction of the trial shown.** For forecast: the first part of the trial that the model sees. |
| `mae.mask.modality` (advanced) | none | text | **Modality to hide.** For modality: which modality to predict. Empty: a random modality on each trial. |
| `mae.n_examples` (advanced) | 16 | whole number, 0 to 200 | **Example trials to save.** Number of test trials whose reconstructions are saved for plotting. |

## Training (`train`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `train.epochs` | 80 (preset) | whole number, 1 to 2000 | **Epochs (at most).** Maximum number of passes over the training trials. Training stops earlier when the validation score stops improving. |
| `train.batch_size` (advanced) | 32 (preset) | whole number, 2 to 2048 | **Batch size.** Trials per training step. Larger batches are faster on a GPU but need more memory. |
| `train.lr` (advanced) | 0.001 (preset) | number, 0 to 1 | **Learning rate.** How large each update step is. Too high: training becomes unstable; too low: it learns slowly. |
| `train.weight_decay` (advanced) | 0.01 (preset) | number, 0 to 1 | **Weight decay.** Pulls weights towards zero to reduce overfitting. |
| `train.warmup_fraction` (advanced) | 0.05 | number, 0 to 0.5 | **Warm-up fraction.** Fraction of training during which the learning rate ramps up from zero. |
| `train.min_lr_ratio` (advanced) | 0.01 | number, 0 to 1 | **Final learning rate (fraction).** The learning rate decays to this fraction of its peak by the end of training. |
| `train.grad_clip` (advanced) | 1.0 (preset) | number, at least 0 | **Gradient clipping.** Limits the size of each update to keep training stable. Empty: no limit. |
| `train.patience` | 15 (preset) | whole number, 1 to 2000 | **Early-stopping patience (epochs).** Stop when the validation score has not improved for this many epochs. The best model is kept. |
| `train.min_epochs` (advanced) | 20 (preset) | whole number, 0 to 2000 | **Minimum epochs.** Early stopping cannot end training before this many epochs. Transformers often improve slowly at first, then quickly. |
| `train.monitor` (advanced) | auto | `auto` / `val_loss` / `val_balanced_accuracy` | **Validation score.** What 'improved' means for early stopping and choosing the best epoch. auto: the validation loss. |
| `train.seed` (advanced) | 0 | whole number, at least 0 | **Training seed.** Seed for weight initialization, batch order and masking. |
| `train.device` | auto | `auto` / `cpu` / `cuda` / `mps` | **Device.** auto: an NVIDIA GPU if there is one, otherwise the Apple GPU, otherwise the CPU. |
| `train.max_minutes` (advanced) | none | number, at least 0 | **Time limit (minutes).** Stop training after this many minutes and keep the best model so far. Empty: no limit. |

## Reference models (`baselines`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `baselines.logistic` (advanced) | on | on / off | **Linear reference model.** Also fit logistic regression on the same trials, as a simple reference the transformer should match or beat. |
| `baselines.permutations` (advanced) | 0 | whole number, 0 to 1000 | **Shuffled-label repeats.** Also refit the logistic-regression reference this many times with shuffled training labels (whole sessions or animals when the label is constant within them). Gives the chance level for your data and a p-value for the transformer's score. Classification only; slow for large datasets. |

## Output (`output`)

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `output.root` | runs | file or folder | **Runs folder.** Each run gets its own folder inside this one. |
| `output.report_by` (advanced) | subject | list | **Report results by.** Trial columns to break results down by (subject and session mean the columns marked with those roles). |
| `output.save_embeddings` (advanced) | on | on / off | **Save trial embeddings.** Save the model's summary vector of every trial, for plotting how trials are organized. |
| `output.overwrite` (advanced) | off | on / off | **Overwrite run folder.** Allow reusing an existing run folder name. Off: an existing folder is never overwritten. |

## Automatic tuning (`hpo`)

Only used by tuning runs (Tune automatically in the app, or prismt.hpo).

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `hpo.n_trials` | 20 | whole number, 1 to 2000 | **Settings to try.** How many combinations of model and training settings automatic tuning tries. |
| `hpo.timeout_minutes` | none | number, at least 0 | **Tuning time budget (minutes).** Stop trying new settings after this long. Empty: no limit. |
| `hpo.epochs_per_trial` (advanced) | none | whole number, 1 to 2000 | **Epochs per tried setting.** Maximum epochs for each tried setting. Empty: the Epochs setting. |
| `hpo.space` (advanced) | default | `default` / `small` | **Search space.** default: learning rate, weight decay, dropout, model size and depth. small: learning rate and dropout only. |
| `hpo.final_seeds` (advanced) | 3 | whole number, 1 to 10 | **Final repeats.** The best settings are retrained this many times with different seeds; the test score is reported as mean and spread. |
| `hpo.workers` (advanced) | 1 | whole number, 1 to 64 | **Parallel workers.** On a cluster: how many jobs search in parallel. |
