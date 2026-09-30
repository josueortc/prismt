function A = load(name)
%LOAD A brain atlas for drawing per-channel maps.
%
%   A = prismt.atlas.load("grid82")    % 82 grid tiles: 41 regions x 2 hemispheres
%   A = prismt.atlas.load("allen52")   % 52 Allen parcels (left/right interleaved)
%   A = prismt.atlas.load("grid41")    % the grid with both hemispheres of region k labelled k
%
%   A.Labels   label image (0 = background, k = channel k) for grid atlases
%   A.Masks    H x W x K logical masks (every atlas; Allen parcels can overlap)
%   A.X, A.Y   tile centroids in image pixels;  A.Hemisphere "L"/"R" per channel
%   A.Name, A.NChannels, A.Description
%   The grid is the schematic tile layout used for widefield data (grid_values.npy in the
%   lab's data), shown in the orientation of the pre-rebuild PRISMT figures.
arguments
    name (1, 1) string {mustBeMember(name, ["grid82", "grid41", "allen52"])}
end
persistent cache
if isempty(cache)
    cache = load(fullfile(fileparts(mfilename('fullpath')), 'private', 'atlases.mat'));
end
switch name
    case "grid82"
        labels = double(cache.grid82_labels);
        A = struct('Name', name, 'NChannels', 82, 'Labels', labels, 'Masks', masksOf(labels, 82), ...
            'X', cache.grid82_x(:), 'Y', cache.grid82_y(:), 'Hemisphere', string(cache.grid82_hemisphere(:)), ...
            'Description', "82 grid tiles (41 regions x 2 hemispheres; channels 2k-1 and 2k are one region)");
    case "grid41"
        labels = ceil(double(cache.grid82_labels) / 2);
        x = cache.grid82_x(:); y = cache.grid82_y(:);
        A = struct('Name', name, 'NChannels', 41, 'Labels', labels, 'Masks', masksOf(labels, 41), ...
            'X', (x(1:2:end) + x(2:2:end)) / 2, 'Y', (y(1:2:end) + y(2:2:end)) / 2, ...
            'Hemisphere', repmat("both", 41, 1), ...
            'Description', "41 grid regions (both hemispheres drawn for each region)");
    case "allen52"
        masks = logical(cache.allen52_masks);
        labels = zeros(size(masks, 1), size(masks, 2));
        for k = size(masks, 3):-1:1
            labels(masks(:, :, k)) = k;         % where parcels overlap, the lower number is drawn
        end
        [x, y] = deal(nan(52, 1));
        for k = 1:52
            [r, c] = find(masks(:, :, k));
            if ~isempty(r), x(k) = mean(c); y(k) = mean(r); end
        end
        A = struct('Name', name, 'NChannels', 52, 'Labels', labels, 'Masks', masks, 'X', x, 'Y', y, ...
            'Hemisphere', repmat(["L"; "R"], 26, 1), ...
            'Description', "52 Allen-atlas parcels, left/right interleaved (21, 22, 49 and 50 are empty)");
end
end

function M = masksOf(labels, K)
M = false([size(labels), K]);
for k = 1:K
    M(:, :, k) = labels == k;
end
end
