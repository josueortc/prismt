function issues = newIssues()
%NEWISSUES Empty list of validation issues (struct array Level/Code/Message/Hint/Field).
issues = struct('Level', {}, 'Code', {}, 'Message', {}, 'Hint', {}, 'Field', {});
end
