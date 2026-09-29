function makeFixtures()
%MAKEFIXTURES Write the MATLAB-side golden fixtures in tests/fixtures.
%
%   Run only when the dataset format version changes. Python's tests read these files and
%   check every value by index, so any disagreement about dimension order, text encoding
%   or metadata types between MATLAB and Python fails loudly.
%     matlab -batch "addpath('matlab','tests/matlab'); makeFixtures" > fixtures.log 2>&1
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
addpath(fullfile(root, 'matlab'));
out = fullfile(root, 'tests', 'fixtures');

% Value-coded array: X(n,r,t,m) = 1000n + 100r + 10t + m, with one planted NaN.
X = valueCoded(5, 4, 6, 2);
X(2, 3, 4, 1) = NaN;
trials = table(categorical(["M1"; "M1"; "M2"; "M2"; "M3"]), [1; 1; 2; 1; 1], ...
    categorical(["early"; "early"; "late"; "late"; "early"], ["early", "late"]), [0; 1; 0; 1; 0], ...
    [true; false; true; true; false], ["café µ"; "a"; "b"; "c"; "d"], ...
    'VariableNames', {'mouse', 'session', 'phase', 'stim', 'hit', 'note'});
ds = prismt.makeDataset(X, trials, SamplingRate=10, TimeZero=-0.2, Event="stimulus onset", ...
    ChannelNames=["Visual L"; "Visual R"; "Frontal L"; "Frontal R"], ChannelX=[1 2 1 2], ChannelY=[2 2 1 1], ...
    ModalityNames=["calcium", "ach"], ModalityUnits=["dF/F", "dF/F"], ...
    Subject="mouse", Session="session", ValueLabels=struct('stim', {{0, "CS-"; 1, "CS+"}}), ...
    Provenance=struct('note', 'golden fixture written by tests/matlab/makeFixtures.m'));
prismt.writeDataset(ds, fullfile(out, 'dataset_v1_matlab.mat'));

% Single modality: MATLAB drops the trailing singleton dimension when saving.
X1 = valueCoded(3, 4, 5, 1);
ds1 = prismt.makeDataset(X1, table(categorical(["A"; "B"; "C"]), 'VariableNames', {'mouse'}), ...
    SamplingRate=10, Subject="mouse", ModalityNames="calcium");
prismt.writeDataset(ds1, fullfile(out, 'dataset_v1_matlab_m1.mat'));
fprintf('Fixtures written to %s\n', out);
end

function X = valueCoded(N, R, T, M)
[n, r, t, m] = ndgrid(1:N, 1:R, 1:T, 1:M);
X = single(1000 * n + 100 * r + 10 * t + m);
end
