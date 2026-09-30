function atomicWrite(file, text)
%ATOMICWRITE Write UTF-8 text through a temporary file, so readers never see half a file.
file = char(file);
tmp = [file '.tmp'];
fid = fopen(tmp, 'w', 'n', 'UTF-8');
if fid < 0
    error('prismt:E_WRITE', 'Cannot write %s.', file);
end
fwrite(fid, unicode2native(char(text), 'UTF-8'));
fclose(fid);
movefile(tmp, file, 'f');
end
