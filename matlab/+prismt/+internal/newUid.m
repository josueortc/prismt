function uid = newUid()
%NEWUID A random 16-character identifier (hex), e.g. for datasets and runs.
%   Uses its own random stream, so it never changes the global rng state.
persistent stream
if isempty(stream)
    stream = RandStream('mt19937ar', 'Seed', mod(floor(now * 86400 * 1000) + feature('getpid'), 2^32)); %#ok<TNOW1>
end
uid = string(lower(dec2hex(randi(stream, [0 15], 1, 16))'));
uid = strjoin(uid, "");
end
