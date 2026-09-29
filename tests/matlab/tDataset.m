classdef tDataset < matlab.unittest.TestCase
    %TDATASET Writing and reading PRISMT datasets from MATLAB.

    properties
        Dir
    end

    methods (TestMethodSetup)
        function tempFolder(tc)
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
        end
    end

    methods (Test)
        function roundTripKeepsEveryValueInPlace(tc)
            X = tDataset.valueCoded(5, 4, 3, 2);
            X(2, 3, 1, 2) = NaN;
            ds = prismt.makeDataset(X, table(categorical(["a"; "a"; "b"; "b"; "c"]), 'VariableNames', {'mouse'}), ...
                SamplingRate=10, Subject="mouse", ModalityNames=["calcium", "ach"]);
            f = prismt.writeDataset(ds, fullfile(tc.Dir, "d.mat"));
            back = prismt.loadDataset(f);
            tc.verifySize(back.X, [5 4 3 2]);
            tc.verifyEqual(back.X, X);
            tc.verifyEqual(back.ModalityNames, ["calcium"; "ach"]);
            tc.verifyEqual(back.Times, (0:2)' / 10, 'AbsTol', 1e-12);
        end

        function singleModalityAndSingleTrialKeepShape(tc)
            for sz = {[5 4 3 1], [1 4 3 2], [3 1 1 1]}
                X = tDataset.valueCoded(sz{1}(1), sz{1}(2), sz{1}(3), sz{1}(4));
                ds = prismt.makeDataset(X, [], SamplingRate=10);
                back = prismt.loadDataset(prismt.writeDataset(ds, fullfile(tc.Dir, "s.mat")));
                tc.verifyEqual([back.N back.R back.T back.M], sz{1});
                tc.verifyEqual(back.X, X);
            end
        end

        function axisOrderIsApplied(tc)
            X = tDataset.valueCoded(5, 4, 3, 1);           % trials x channels x time
            Y = permute(X, [1 3 2]);                        % trials x time x channels
            ds = prismt.makeDataset(Y, [], AxisOrder="trials,time,channels", SamplingRate=10);
            tc.verifyEqual(ds.X, X);
        end

        function metadataTypesAndLabelsSurvive(tc)
            X = tDataset.valueCoded(4, 2, 2, 1);
            T = table(["M1"; "M1"; "M2"; ""], [0; 1; 0; NaN], [true; false; true; true], ...
                categorical(["early"; "late"; "late"; "early"], ["early", "late"]), ["café µ"; "x"; "y"; "z"], ...
                'VariableNames', {'mouse', 'stim', 'hit', 'phase', 'lick latency'});
            ds = prismt.makeDataset(X, T, SamplingRate=10, Subject="mouse", ...
                ValueLabels=struct('stim', {{0, "CS-"; 1, "CS+"}}));
            back = prismt.loadDataset(prismt.writeDataset(ds, fullfile(tc.Dir, "m.mat")));
            tc.verifyClass(back.Trials.mouse, 'categorical');
            tc.verifyTrue(isundefined(back.Trials.mouse(4)));
            tc.verifyEqual(back.Trials.stim, [0; 1; 0; NaN]);
            tc.verifyClass(back.Trials.hit, 'logical');
            tc.verifyEqual(categories(back.Trials.phase), {'early'; 'late'});
            tc.verifyEqual(string(back.Trials.("lick latency")(1)), "café µ");
            tc.verifyEqual(back.ValueLabels.Label, ["CS-"; "CS+"]);
            tc.verifyEqual(back.Subject, "mouse");
        end

        function fileLayoutIsWhatPythonExpects(tc)
            X = tDataset.valueCoded(5, 4, 3, 2);
            f = prismt.writeDataset(prismt.makeDataset(X, [], SamplingRate=10), fullfile(tc.Dir, "l.mat"));
            info = h5info(f);
            names = string({info.Datasets.Name});
            tc.verifyTrue(all(ismember(["X", "meta_json", "prismt_format", "prismt_version"], names)));
            xinfo = info.Datasets(names == "X");
            tc.verifyEqual(xinfo.Datatype.Class, 'H5T_FLOAT');
            tc.verifyEqual(xinfo.Datatype.Size, 4);                % single precision
            tc.verifyFalse(isfile(f + ".partial"));
            meta = jsondecode(native2unicode(reshape(h5read(f, '/meta_json'), 1, []), 'UTF-8'));
            tc.verifyEqual(meta.shape(:)', [5 4 3 2]);
            tc.verifyGreaterThanOrEqual(numel(meta.orientation_probes), 3);
        end

        function infiniteValuesAreRejected(tc)
            X = tDataset.valueCoded(3, 2, 2, 1);
            X(2, 1, 2) = Inf;
            tc.verifyError(@() prismt.makeDataset(X, [], SamplingRate=10), 'prismt:E_DATA_INF');
        end

        function duplicateChannelNamesAreRejected(tc)
            X = tDataset.valueCoded(3, 2, 2, 1);
            tc.verifyError(@() prismt.makeDataset(X, [], SamplingRate=10, ChannelNames=["a", "a"]), ...
                'prismt:E_DATA_CHANNELS');
        end

        function trialTableHeightMustMatch(tc)
            X = tDataset.valueCoded(3, 2, 2, 1);
            T = table([1; 2], 'VariableNames', {'stim'});
            tc.verifyError(@() prismt.makeDataset(X, T, SamplingRate=10), 'prismt:E_DATA_COLUMNS');
        end

        function perTrialTimeSeriesColumnIsRejectedWithAHint(tc)
            X = tDataset.valueCoded(3, 2, 2, 1);
            T = table(rand(3, 5), 'VariableNames', {'runSpeed'});
            err = captureError(@() prismt.makeDataset(X, T, SamplingRate=10));
            tc.verifyEqual(err.identifier, 'prismt:E_DATA_COLUMNS');
            tc.verifySubstring(err.message, 'modality');
        end

        function modalityChannelSetsAreChecked(tc)
            X = tDataset.valueCoded(3, 2, 2, 2);
            tc.verifyError(@() prismt.makeDataset(X, [], SamplingRate=10, ModalityChannels={1, [1 3]}), ...
                'prismt:E_DATA_MODALITIES');
        end

        function subsetKeepsMetadataAligned(tc)
            [ds, ~] = prismt.demo.makeSyntheticDataset(Profile="tiny");
            keep = ds.Trials.phase == "late";
            sub = ds.subset(keep);
            tc.verifyEqual(sub.N, nnz(keep));
            tc.verifyTrue(all(sub.Trials.phase == "late"));
            tc.verifyEqual(sub.X, ds.X(keep, :, :, :));
        end

        function loadingANonPrismtFileExplainsWhatToDo(tc)
            T = table([1; 2]);                                       %#ok<NASGU>
            f = fullfile(tc.Dir, "raw.mat");
            save(f, 'T', '-v7.3');
            err = captureError(@() prismt.loadDataset(f));
            tc.verifyEqual(err.identifier, 'prismt:E_DATA_NOT_PRISMT');
            tc.verifySubstring(err.message, 'importData');
        end
    end

    methods (Static)
        function X = valueCoded(N, R, T, M)
            [n, r, t, m] = ndgrid(1:N, 1:R, 1:T, 1:M);
            X = single(1000 * n + 100 * r + 10 * t + m);
        end
    end
end
