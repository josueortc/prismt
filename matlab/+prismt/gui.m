function app = gui(opts)
%GUI Open the PRISMT app: set up Python, load data, choose what to learn, train, look at results.
%
%   prismt.gui()                      % or run_prismt_gui from the PRISMT folder
%   app = prismt.gui(Visible="off")   % no window (scripts and tests); app.Controller does the work
%
%   Everything the app does can also be scripted; see prismt.train, prismt.check and the
%   tutorial (matlab/examples/prismt_tutorial.m). Tab > Results > "Export as MATLAB script"
%   writes the script for any run.
arguments
    opts.Visible (1, 1) string {mustBeMember(opts.Visible, ["on", "off"])} = "on"
    opts.Controller = []
    opts.Dialogs = []
end
if string(version('-release')) < "2021a"     % e.g. "2020b" < "2021a" (text order is release order)
    error('prismt:release', 'PRISMT needs MATLAB R2021a or newer (this is R%s).', version('-release'));
end
app = prismt.app.App(Visible=opts.Visible, Controller=opts.Controller, Dialogs=opts.Dialogs);
end
