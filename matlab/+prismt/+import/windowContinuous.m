function W = windowContinuous(Xtc, L, stride)
%WINDOWCONTINUOUS Cut a continuous time x channels recording into windows.
%   W = prismt.import.windowContinuous(Xtc, L, stride) returns [nWindows x channels x L].
%   Window w holds samples start_w .. start_w+L-1 of every channel. (The pre-rebuild CDKL5
%   scripts used reshape(X, [n, 30, R]), which interleaved samples from across the whole
%   recording into each "trial".)
[nT, R] = size(Xtc);
if nT < L
    W = zeros(0, R, L);
    return
end
starts = 1:stride:(nT - L + 1);
idx = starts + (0:L - 1)';                           % L x nWindows sample indices
W = permute(reshape(Xtc(idx, :), L, numel(starts), R), [2 3 1]);
end
