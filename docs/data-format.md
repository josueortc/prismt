# The PRISMT dataset file (format 1)

PRISMT trains on one kind of file: a **PRISMT dataset**. It holds every trial of your
experiment as an array of size **trials × channels × time × modalities**, plus a
description of the channels, the time axis and what happened on each trial.

You rarely write this file by hand. In MATLAB:

- **Data tab → Import…** converts common lab formats (a table `T` with one row per session,
  `processed_data` structs, numbered variables, CDKL5 structs, or variables in your
  workspace).
- **`prismt.makeDataset`** builds one from your own arrays in one line.
- **`prismt.demo.makeSyntheticDataset`** makes demo data with known structure.

Then **`prismt.writeDataset(ds, "mydata_prismt.mat")`** saves it, and
**`prismt.loadDataset`** reads it back.

## Words used here

| Term | Meaning | Examples |
|---|---|---|
| **trial** | one repetition of your task; one row of metadata | a stimulus presentation |
| **channel** | one thing recorded over time | a cortical region, an electrode, a grid tile, a behavior variable (speed, pupil, a paw coordinate), a physiological signal |
| **time** | samples within a trial, the same for every trial | 41 frames at 10 Hz, from −1.1 s to 2.9 s |
| **modality** | a kind of signal recorded on (some of) the channels | calcium, acetylcholine, running speed |
| **subject** | the animal or participant a trial came from | mouse `HB051` |
| **channel group** | an optional label per channel | `left`/`right`, a brain area, `arm`/`eye` for sensors |
| **session** | the recording session a trial came from | day 3 of learning |

Channels of different modalities do not have to match: a dataset can have 82 calcium
channels and 3 behavior channels. Each modality lists which channels exist for it
(`ModalityChannels`), and the rest are filled with `NaN`. The padding makes the file larger
(82 calcium + 3 behavior channels are stored as 85 channels for each modality) but costs no
training time: the model only sees the channels each modality actually has.

## Making a dataset from your own arrays (MATLAB)

```matlab
X = dff;                 % [trials x channels x time], e.g. 5000 x 82 x 41
info = table(mouse, session, phase, stim, response);   % one row per trial
ds = prismt.makeDataset(X, info, ...
    SamplingRate=10, TimeZero=-1.1, Event="stimulus onset", ...
    ModalityNames="calcium", ModalityUnits="dF/F", ...
    Subject="mouse", Session="session", ...
    ValueLabels=struct('stim', {{0, "CS-"; 1, "CS+"}}));
prismt.writeDataset(ds, "mydata_prismt.mat");
```

- If the dimensions of `X` are in another order, say so with `AxisOrder`, e.g.
  `AxisOrder="trials,time,channels"`. PRISMT never guesses the order.
- To add a second signal (e.g. ACh), stack along the fourth dimension:
  `X = cat(4, calcium, ach)` with `ModalityNames=["calcium","ach"]`.
- Use `NaN` for anything missing. PRISMT never trains on, or scores, a missing value.

**Always mark the subject.** Trials from the same subject are not independent. With
`Subject` set, PRISMT keeps subjects separate between training and testing, so accuracy
means "works on a new subject", not "recognized a subject it has seen".

**Recordings with different channels.** Make one dataset per recording and join them with
`prismt.combineDatasets({ds1, ds2, ...})`: channels and signals are matched by name, a
channel a recording lacks is missing (`NaN`) for its trials, and trial columns are joined.
Different time axes need `Resample=true`. The Data tab's *Add dataset...* does the same.

**Signals that are not brain activity** are handled the same way: give them a name, units and
a kind (`ModalityKinds`, any short text such as `neural`, `behavior`, `physiology`,
`stimulus`). Each channel of each signal is normalized separately (on training trials), so
signals in different units can be combined.

## The file itself

The file is a MATLAB **v7.3** `.mat` (HDF5) with four variables (five with an atlas):

| Variable | MATLAB class and size | Content |
|---|---|---|
| `prismt_format` | `uint8`, the text `prismt.dataset` | format tag |
| `prismt_version` | `double` scalar, `1` | format version |
| `X` | `single` `[N R T M]` | the data; `NaN` = missing |
| `meta_json` | `uint8` row, UTF-8 text | JSON description, below |
| `atlas_image` *(optional)* | `uint16` `[H W]` | label image for maps: pixel value *k* = channel *k*, 0 = background |

MATLAB drops trailing singleton dimensions when it saves, so a single-modality `X` is
stored as `[N R T]`. `meta_json.shape` is always the full size.

### `meta_json`

```json
{
  "format": "prismt.dataset", "version": 1,
  "dataset_uid": "3f9c0b1e2d7a4c55",
  "created": "2026-09-29T16:37:00Z", "created_by": "prismt-matlab 0.1.0 (MATLAB 2025a)",
  "shape": [N, R, T, M],
  "channels": {
    "names": ["ch01", "..."],
    "x": [1, 2, "..."] , "y": [1, 1, "..."],
    "hemisphere": ["L", "R", "..."],
    "groups": ["left", "right", "..."],
    "atlas": "grid82"
  },
  "modalities": {
    "names": ["calcium", "ach"], "units": ["dF/F", "dF/F"],
    "kinds": ["neural", "neural"],
    "channels": [[1, 2, "..."], [1, 2, "..."]]
  },
  "time": {"times_s": [-1.1, -1.0, "..."], "fs_hz": 10, "t0_s": -1.1, "event": "stimulus onset"},
  "trials": {
    "columns": [
      {"name": "mouse", "type": "categorical", "values": ["HB051", "..."], "categories": ["HB051", "..."]},
      {"name": "stim", "type": "numeric", "values": [0, 1, "..."],
       "labels": [{"value": 0, "label": "CS-"}, {"value": 1, "label": "CS+"}]},
      {"name": "hit", "type": "bool", "values": [true, false, "..."]}
    ],
    "roles": {"subject": "mouse", "session": "session"},
    "uid": null
  },
  "orientation_probes": [{"index": [0, 0, 0, 0], "value": 0.12}, "..."],
  "provenance": {"source_files": ["..."], "import_recipe": {}}
}
```

Rules:

- **Lists are always JSON arrays.** Readers also accept a scalar where a one-element list
  is expected, because MATLAB's `jsonencode` writes `[5]` as `5`.
- **Missing values are `null`**: in numeric columns, in text columns, and for optional
  fields.
- `channels.x`, `y`, `hemisphere` and `atlas` are optional (`null`). They are only used to
  draw maps.
- `modalities.kinds` is `neural`, `behavior` or `other`. `modalities.channels` holds
  1-based channel numbers. Values outside a modality's channels are ignored and should be
  `NaN`.
- `time.times_s` gives the time of every sample, strictly increasing. `fs_hz`/`t0_s` are
  written as well when sampling is uniform.
- **Trial columns** are an array of objects, so column names can be any text. `type` is
  `numeric`, `categorical` (text) or `bool`. `labels` gives names for numeric codes.
- **`roles`** name the columns that identify the subject and the session. Session numbers
  may repeat across subjects; PRISMT combines the two into a unique session key.
- **`orientation_probes`** hold a few values of `X` with their **0-based** positions. After
  restoring the dimension order, readers check them and refuse the file if any differs.
  A shape check alone cannot catch swapped channel and time axes when both have the same
  size; the probes can.

### Reading it outside MATLAB

h5py reports a MATLAB `[N R T M]` array reversed, as `(M, T, R, N)`, possibly without the
leading `M` when `M = 1`. `prismt.io.read_dataset` reverses the axes, restores the declared
shape and checks the probes:

```python
from prismt.io import read_dataset
ds = read_dataset("mydata_prismt.mat")   # ds.X is float32 (N, R, T, M)
```

From Python you can also write datasets. `prismt.io.write_dataset` produces the same file,
readable with MATLAB's `load`. It can also write an `.npz` with `X` in `(N, R, T, M)` order
and `meta_json` as text.

## Checking a file

```text
python -m prismt validate mydata_prismt.mat          # human-readable
python -m prismt validate mydata_prismt.mat --json   # what the MATLAB app reads
```

Every problem is reported with the field it concerns and how to fix it, for example
`X contains 3 infinite value(s); the first is X(12,5,3,1).`
