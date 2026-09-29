function err = captureError(fcn)
%CAPTUREERROR Run fcn and return the MException it throws ([] if it does not throw).
err = [];
try
    fcn();
catch err
end
end
