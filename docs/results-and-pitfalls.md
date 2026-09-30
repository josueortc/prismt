# Reading the results, and the pitfalls PRISMT guards against

A transformer can reach a high score for the wrong reason: it recognizes the animal, the
session, or where data are missing, instead of the condition you care about. This page says
what each number means, what PRISMT does to keep it honest, and what is still up to you.

## What the score is

Every run splits the trials three ways:

| Part | Used for |
|---|---|
| **Training** | Fitting the model. |
| **Validation** | Deciding when to stop and which epoch to keep (and, when tuning, which settings win). |
| **Test** | The reported score. Test trials are evaluated **once**, after training, and never influence any choice. |

The normalization (z-scoring) is also fitted on training trials only.

**Classification** reports **balanced accuracy**, the average of the per-class accuracies,
so a model that always guesses the most common class scores 1/number of classes (0.5 for
two classes) however unbalanced they are. Accuracy, macro-F1, AUROC and the confusion
matrix are in the run's `metrics.json`.

**Masked autoencoder** reports the **R² of the hidden values**: 1 is perfect, 0 is no better
than predicting the average, and below 0 is worse than that. Each hiding pattern (random,
whole channels, forecast, whole signal) is scored separately.

## Compare with the reference models

Each result comes with simple models evaluated on the **same test trials**:

- classification: chance, always guessing the most common class, and logistic regression
  on the trial's values;
- autoencoder: the trial-average (PSTH) of each channel and time, the mean of the visible
  values of the same channel, the last visible value, and interpolation between visible
  values.

A transformer score is only interesting when it beats these. When logistic regression does
as well or better (common with a few thousand trials), that is itself a finding: the
information is there, but no complex model is needed to read it out. The quick preset is
also deliberately small; train longer or use the Standard preset before concluding the
transformer cannot do better.

## Testing on new animals

Trials of one animal (and one session) are not independent: they share the animal's
anatomy, the imaging quality, the day's arousal. A model tested on held-out *trials* of
animals it trained on can score high by recognizing each animal. Tested on *new animals*,
the score says whether the effect generalizes.

- **Test on: Automatic** (the default) tests on new animals when there are at least three,
  else on new sessions, else on held-out trials, and the Checks say which it chose.
- If a label is the same for all trials of a session (like `phase`: a whole session is
  early or late), testing on held-out trials or sessions of the same animals would let the
  model recognize sessions. PRISMT refuses this (error `E_LEAKY_SPLIT`, with the fix). It
  can be overridden (`split.allow_leaky`), but the run is then flagged `leaky_split` and
  its scores are marked as inflated.
- Without a subject column, PRISMT warns that the score only shows the model works on
  held-out trials, not on new animals.

## Few animals: cross-validation

With fewer than 10 animals, one test split would contain only one or two of them, so
PRISMT uses cross-validation instead: with 6–9 animals, 5 folds; with 3–5 animals, *leave
one out*, where each animal is the test set once. The headline score pools the test
predictions of all folds; the summary also gives the mean ± sd across folds.

Look at **Accuracy per animal**: one dot per animal shows whether the effect holds in most
animals or comes from one or two. With four animals, a score of 0.7 can mean "every
animal about 0.7" or "two at 0.9 and two at chance"; these support very different
conclusions.

## Confounds

**Trial info** (Data tab) counts trials for two columns. If, for example, every late
session also has a different lick rate, a model classifying `phase` may learn licking.
PRISMT's check warns when another column almost perfectly predicts the label
(`W_CONFOUND`, Cramér's V ≥ 0.9) and when classes differ in how much data is missing
(`W_MISSING_BY_CLASS`), since a model can tell classes apart by where values are missing.
What to do about a confound depends on the question: balance it, test within one level of
it (a trial filter), or report it.

Behavior recorded as an extra signal (e.g. running speed) is used by the model like any
other input. To ask whether the neural signal alone carries the information, train once
with and once without it (`selection.modalities`).

## Reading autoencoder results

- **Predictability per channel** (random or whole-channel hiding) shows which channels can
  be predicted from the others. A channel that is pure noise, or badly imaged, is not
  predictable; on the demo data the planted noise channel shows up this way.
- **Gain over baseline per channel** subtracts the best reference model, so a bright
  channel is one where the transformer uses information the trial average does not have.
- A **negative forecast R²** is common for a model trained with random hiding: predicting
  the second half of a trial from the first is a different, harder task. Train with
  *Hide: The future* to ask that question.
- **Predictability by condition** (e.g. early vs late) is computed after the fact from
  per-trial sums, with intervals from resampling sessions. It is exploratory: the model
  was not trained to compare conditions, and several conditions compared this way invite
  false positives.

## Tuning

Automatic tuning compares settings on validation trials only, retrains the best settings
with several seeds, and tests each once (mean ± sd over seeds). Choosing the best of many
tries on the test trials would overstate the score; PRISMT never does that. Tuning uses the
first fold's split, so with cross-validation the tuned settings are then applied to all
folds by the final retraining.

## Chance with few trials

With few test trials, a score well above 1/number of classes can still happen by chance.
`baselines.permutations` (e.g. 200) refits logistic regression with shuffled training
labels, on the same split, to give the scores reached when there is no effect. When the
label is the same for whole sessions or animals, whole sessions or animals are shuffled, so
the comparison keeps the data's structure. The run then reports the chance level for your
data (the 95th percentile is drawn in *Score vs reference models*) and a p-value for the
transformer's score. It is slow for large datasets, so it is off by default.

## Reproducing a run

A run folder contains everything needed to repeat it: `config.json` (the resolved
settings), `run_info.json` (versions, device, git commit), `splits.csv` (which trial was
in which part), and `launch.sh` / `launch.cmd` (the exact command). *Export as MATLAB
script* writes the settings as a script. Results can differ slightly between devices (CPU,
Apple GPU, NVIDIA GPU) because their arithmetic differs; the splits and hidden-value masks
do not, since they are drawn on the CPU from fixed seeds.

## What the previous version got wrong

The code in `legacy/` had problems that the rebuild fixes by design; they are worth
knowing when reading results produced with it:

- array orientation was guessed from sizes, which could silently build wrong labels
  (e.g. classifying by stimulus actually learned phase);
- the "causal" attention mask was applied after the softmax, so it did not prevent
  attending to the future;
- the validation split chose the checkpoint, stopped training and produced the reported
  score, with no separate test split;
- cutting continuous CDKL5 recordings into trials with `reshape` interleaved samples from
  across the recording into each trial.

See [legacy/README.md](../legacy/README.md) for the full list.
