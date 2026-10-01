classdef tChannels < matlab.unittest.TestCase
    %TCHANNELS Datasets whose channels differ: combining, importing, channel groups, and
    %signals that are not brain activity.

    properties
        Dir
    end

    methods (TestMethodSetup)
        function setup(tc)
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
        end
    end

    methods (Test)
        function combineMatchesChannelsAndSignalsByName(tc)
            % Recording A: 3 electrodes + pupil; recording B: 2 of the electrodes, a new one, no pupil.
            XA = cat(4, 1 + zeros(4, 3, 5), 2 + zeros(4, 3, 5));          % trials x chan x time x signal
            A = prismt.makeDataset(XA, table(["s1";"s1";"s2";"s2"], [0;1;0;1], 'VariableNames', {'subject','cond'}), ...
                SamplingRate=100, ChannelNames=["e1","e2","e3"], ChannelGroups=["motor","motor","visual"], ...
                ModalityNames=["lfp","pupil"], ModalityKinds=["neural","behavior"], ModalityChannels={(1:3)', 3}, ...
                Subject="subject");
            XB = 5 + zeros(3, 3, 5);
            B = prismt.makeDataset(XB, table(["s3";"s3";"s3"], [1;0;1], ["x";"y";"z"], 'VariableNames', {'subject','cond','extra'}), ...
                SamplingRate=100, ChannelNames=["e2","e3","e9"], ModalityNames="lfp", ModalityKinds="neural", Subject="subject");
            [ds, notes] = prismt.combineDatasets({A, B}, Source="recording", Names=["A","B"]);
            tc.verifyEqual(ds.ChannelNames, ["e1";"e2";"e3";"e9"]);
            tc.verifyEqual(ds.ModalityNames, ["lfp";"pupil"]);
            tc.verifyEqual(ds.ModalityKinds, ["neural";"behavior"]);
            tc.verifyEqual(ds.ModalityChannels{1}, (1:4)');
            tc.verifyEqual(ds.ModalityChannels{2}, 3);
            tc.verifyEqual(ds.N, 7);
            lfp = ds.X(:, :, 1, 1);
            tc.verifyEqual(lfp(1, :), single([1 1 1 NaN]));
            tc.verifyEqual(lfp(5, :), single([NaN 5 5 5]), "B's channels land on their names");
            tc.verifyTrue(all(isnan(ds.X(5:7, 3, :, 2)), 'all'), "B has no pupil");
            tc.verifyEqual(ds.ChannelGroups(1:3), ["motor";"motor";"visual"]);
            tc.verifyEqual(string(ds.Trials.recording), ["A";"A";"A";"A";"B";"B";"B"]);
            tc.verifyTrue(all(ismissing(ds.Trials.extra(1:4))));
            tc.verifyEqual(ds.Subject, "subject");
            tc.verifyTrue(any(contains(notes, "lacks")));
            tc.verifyEmpty(ds.validate());
            f = prismt.writeDataset(ds, fullfile(tc.Dir, "combined.mat"));
            back = prismt.loadDataset(f);
            tc.verifyEqual(back.ChannelGroups, ds.ChannelGroups);
            tc.verifyEqual(back.ModalityKinds, ["neural";"behavior"]);
        end

        function combineNeedsTheSameTimesUnlessResampled(tc)
            A = prismt.makeDataset(repmat(reshape(0:9, 1, 1, 10), 2, 1), [], SamplingRate=10);        % 0..0.9 s
            B = prismt.makeDataset(repmat(reshape(0:4, 1, 1, 5), 2, 1), [], SamplingRate=5);          % 0..0.8 s
            err = captureError(@() prismt.combineDatasets({A, B}));
            tc.verifyEqual(string(err.identifier), "prismt:E_COMBINE_TIME");
            tc.verifySubstring(string(err.message), "Resample=true");
            ds = prismt.combineDatasets({A, B}, Resample=true);
            tc.verifyEqual(ds.T, 9, "limited to the common range 0-0.8 s");
            tc.verifyEqual(squeeze(ds.X(3, 1, :))', single(0:0.5:4), 'AbsTol', 1e-5, "B interpolated onto A's times");
        end

        function importAcceptsSessionsWithDifferentChannels(tc)
            dff = {rand(3, 4, 6); rand(2, 6, 6)};
            T = table(dff, {[0;1;0]; [1;0]}, {'m1'; 'm2'}, 'VariableNames', {'dff', 'cond', 'mouse'});
            f = fullfile(tc.Dir, "byposition.mat");
            save(f, 'T');
            [ds, rep] = prismt.importData(f);
            tc.verifyEqual(ds.R, 6);
            tc.verifyTrue(all(isnan(ds.X(1:3, 5:6, :, 1)), 'all'), "session 1 lacks channels 5-6");
            tc.verifyTrue(any(contains(rep.notes, "matched by position")));
            tc.verifyEqual(ds.ModalityNames, "dff", "signals keep the name they have in the file");

            names = {["a","b","c","d"]; ["c","d","e","f","g","h"]};
            T.channelNames = names;
            f = fullfile(tc.Dir, "byname.mat");
            save(f, 'T');
            [ds, rep] = prismt.importData(f);
            tc.verifyEqual(ds.ChannelNames, ["a";"b";"c";"d";"e";"f";"g";"h"]);
            tc.verifyEqual(ds.X(4, 3, :, 1), single(dff{2}(1, 1, :)), "session 2's first channel is c");
            tc.verifyTrue(any(contains(rep.notes, "matched by name")));
            tc.verifyFalse(ismember("channelNames", string(ds.Trials.Properties.VariableNames)));
        end

        function signalsGivenByNameOnTheSameTrials(tc)
            N = 12; T = 7;
            neural = rand(N, 5, T); pupil = rand(N, T); speed = rand(N, 1, T);
            trials = table(repelem(["m1"; "m2"; "m3"], 4), repmat([0; 1], 6, 1), 'VariableNames', {'mouse', 'go'});
            ds = prismt.makeDataset(struct('neural', neural, 'pupil', pupil, 'speed', speed), trials, ...
                SamplingRate=30, Subject="mouse", ModalityKinds=["neural", "behavior", "behavior"]);
            tc.verifyEqual(ds.ModalityNames, ["neural"; "pupil"; "speed"]);
            tc.verifyEqual(ds.ChannelNames, ["neural_01"; "neural_02"; "neural_03"; "neural_04"; "neural_05"; "pupil"; "speed"]);
            tc.verifyEqual(ds.ModalityChannels{1}, (1:5)');
            tc.verifyEqual(ds.ModalityChannels{2}, 6);
            tc.verifyEqual(squeeze(ds.X(3, 6, :, 2)), single(pupil(3, :))', 'AbsTol', 1e-6);
            tc.verifyTrue(all(isnan(ds.X(:, 1:5, :, 2)), 'all'), "pupil has no neural channels");
            tc.verifyEmpty(ds.validate());
            err = captureError(@() prismt.makeDataset(struct('a', rand(N, 2, T), 'b', rand(N, T + 1)), trials));
            tc.verifySubstring(string(err.message), "share trials and time points");
        end

        function anyKindOfSignalIsAccepted(tc)
            ds = prismt.makeDataset(rand(6, 2, 3), table((1:6)' > 3, 'VariableNames', {'go'}), ...
                ModalityNames="heart", ModalityKinds="physiology", ModalityUnits="bpm", SamplingRate=1);
            issues = ds.validate();
            tc.verifyFalse(any(string({issues.Level}) == "error"));
            ds2 = prismt.makeDataset(rand(6, 2, 3), [], SamplingRate=1);
            tc.verifyEqual(ds2.ModalityKinds, "signal");
        end
    end
end
