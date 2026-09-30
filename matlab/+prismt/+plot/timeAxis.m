function [t, label] = timeAxis(ds)
%TIMEAXIS Time in seconds and an axis label mentioning the event.
t = ds.Times(:)';
label = "Time (s)";
if strlength(ds.Event), label = "Time from " + ds.Event + " (s)"; end
end
