function setLimits(ax, lim)
%SETLIMITS Color limits that work on R2021a (caxis) and newer releases (clim).
if ~all(isfinite(lim)) || lim(2) <= lim(1), return; end
if exist('clim', 'file') || exist('clim', 'builtin')
    clim(ax, lim);
else
    caxis(ax, lim); %#ok<CAXIS> compat-ok
end
end
