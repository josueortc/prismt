function s = shellQuote(s, style)
%SHELLQUOTE Quote one argument for a POSIX shell ('posix') or Windows cmd ('cmd').
%   POSIX single quotes make every character literal (including $, *, [ ] and spaces,
%   which zsh would otherwise expand).
s = string(s);
if style == "cmd"
    s = """" + replace(s, """", """""") + """";
else
    s = "'" + replace(s, "'", "'\''") + "'";
end
end
