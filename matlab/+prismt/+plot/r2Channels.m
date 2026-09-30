function v = r2Channels(ax, R, opts)
%R2CHANNELS How predictable each channel is from the rest (R² of its hidden values).
%
%   prismt.plot.r2Channels(ax, R)                              % training mask, first modality
%   prismt.plot.r2Channels(ax, R, Mask="channel_0.2", Modality="ach", VersusBaseline=true)
%   Drawn on the dataset's atlas or channel positions when the run has them, else as bars.
%   VersusBaseline shows the gain over the best simple baseline (positive = model better).
arguments
    ax
    R struct
    opts.Mask (1, 1) string = ""
    opts.Modality = 1
    opts.VersusBaseline (1, 1) logical = false
    opts.Atlas (1, 1) string = ""
    opts.X double = []
    opts.Y double = []
end
M = R.Mat;
si = 1;
if strlength(opts.Mask), si = find(M.mask_names == opts.Mask, 1); end
m = opts.Modality;
if ~isnumeric(m), m = find(M.modality_names == string(m), 1); end
pairs = M.pairs;
rows = find(pairs(:, 2) == m);
R2 = M.r2_by_pair(si, rows);
lab = "R² of hidden values";
if opts.VersusBaseline
    base = squeeze(max(M.baseline_r2_by_pair(si, :, rows), [], 2))';
    R2 = R2 - base; lab = "R² gain over best baseline";
end
v = nan(numel(M.channel_names), 1);
v(pairs(rows, 1)) = R2;
atlas = opts.Atlas;
if strlength(atlas) == 0 && isfield(R, 'Config') && ~isempty(R.Config)
    atlas = datasetAtlas(R);
end
ttl = replace(M.mask_names(si), "_", " ") + " — " + M.modality_names(m);
if strlength(atlas) && prismt.atlas.load(atlas).NChannels == numel(v)
    prismt.plot.channelMap(ax, v, Atlas=atlas, Title=ttl, Label=lab);
elseif ~isempty(opts.X)
    prismt.plot.channelMap(ax, v, X=opts.X, Y=opts.Y, Names=M.channel_names, Title=ttl, Label=lab);
else
    prismt.plot.channelMap(ax, v, Names=M.channel_names, Title=ttl, Label=lab);
end
end

function a = datasetAtlas(R)
a = "";
try
    f = string(R.Config.dataset.path);
    if isfile(f)
        meta = jsondecode(native2unicode(reshape(h5read(f, '/meta_json'), 1, []), 'UTF-8'));
        if isfield(meta.channels, 'atlas') && ischar(meta.channels.atlas), a = string(meta.channels.atlas); end
    end
catch
end
end
