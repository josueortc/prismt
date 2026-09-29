function file = writeDataset(ds, file)
%WRITEDATASET Save a PRISMT dataset so that PRISMT (Python) can train on it.
%
%   file = prismt.writeDataset(ds, file)
%
%   Writes a MATLAB v7.3 .mat file with the variables prismt_format, prismt_version, X
%   (single, trials x channels x time x modalities) and meta_json (a UTF-8 JSON
%   description). You can load it back with prismt.loadDataset, or plain load().
%   The file is written under a temporary name and renamed when complete, so an
%   interrupted save never leaves a half-written dataset behind.
arguments
    ds (1, 1) prismt.Dataset
    file (1, 1) string
end
prismt.internal.throwIfErrors(ds.validate(), "The dataset could not be written");
[folder, ~, ext] = fileparts(file);
if ext == ""
    file = file + ".mat";
end
if strlength(folder) > 0 && ~isfolder(folder)
    mkdir(folder);
end
if strlength(ds.Uid) == 0
    ds.Uid = prismt.internal.newUid();
end
S.prismt_format = uint8('prismt.dataset');
S.prismt_version = double(prismt.version().Formats.dataset);
S.X = single(ds.X);
S.meta_json = unicode2native(jsonencode(prismt.internal.datasetMeta(ds)), 'UTF-8');
if ~isempty(ds.AtlasImage)
    S.atlas_image = uint16(ds.AtlasImage);
end
tmp = file + ".partial";
cleanup = onCleanup(@() deleteIfExists(tmp));
save(tmp, '-struct', 'S', '-mat', '-v7.3', '-nocompression');
movefile(tmp, file, 'f');
end

function deleteIfExists(f)
if isfile(f)
    delete(f);
end
end
