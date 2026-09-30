function [py, report, tried] = findPython(opts)
%FINDPYTHON Find a Python that can run PRISMT, and remember it.
%
%   [py, report, tried] = prismt.env.findPython()
%   py is "" when none works; tried lists every candidate with its doctor report.
arguments
    opts.Save (1, 1) logical = true
end
py = "";
report = [];
tried = struct('python', {}, 'ok', {}, 'message', {});
for c = prismt.env.candidates()'
    r = prismt.env.doctor(c);
    msg = "";
    if ~r.ok
        bad = r.checks(~[r.checks.ok]);
        msg = string(bad(1).message);
    end
    tried(end + 1) = struct('python', c, 'ok', r.ok, 'message', msg); %#ok<AGROW>
    if r.ok
        py = c;
        report = r;
        if opts.Save
            prismt.internal.settings("python", char(c));
        end
        return
    end
end
end
