function p = emptyParts()
%EMPTYPARTS The common intermediate form every legacy reader returns.
%   signals      struct: name -> cell(nSessions,1) of [trials x channels x time] arrays
%   behavior     struct: name -> cell(nSessions,1) of [trials x time] arrays
%   trialCols    struct: name -> cell(nSessions,1) of per-trial column vectors
%   sessionCols  struct: name -> nSessions x 1 values (one per session)
%   sessionNames nSessions x 1 string (a unique name per session)
%   channelNames {} or cell(nSessions,1) of channel names per session (when the file has
%                them; sessions may then have different channels, matched by name)
%   fs, t0       sampling rate and time of the first sample ([] if unknown); event: what time 0 is
%   labels       struct: column -> {value, label; ...}
%   atlasHint    e.g. "grid82";  notes: what the reader assumed
p = struct('signals', struct(), 'behavior', struct(), 'trialCols', struct(), 'sessionCols', struct(), ...
    'sessionNames', strings(0, 1), 'channelNames', {{}}, 'fs', [], 't0', [], 'event', "", 'labels', struct(), 'atlasHint', "", 'notes', strings(0, 1));
end
