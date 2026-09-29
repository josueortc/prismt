classdef tSynthetic < matlab.unittest.TestCase
    %TSYNTHETIC The MATLAB implementation of the synthetic recipe.

    methods (Test)
        function structureMatchesTheSharedFixture(tc)
            root = prismt.internal.repoRoot();
            golden = jsondecode(fileread(fullfile(root, 'tests', 'fixtures', 'synthetic_structure_v1.json'))).profiles;
            for profile = ["tiny", "fast", "tutorial"]
                st = prismt.demo.syntheticStructure(profile);
                want = golden.(profile);
                tc.verifyEqual(st.shape, want.shape(:)', profile);
                tc.verifyEqual(st.loadings, want.loadings, 'AbsTol', 1e-6, profile);
                tc.verifyEqual(st.kernel, want.kernel(:)', 'AbsTol', 1e-6, profile);
                tc.verifyEqual(st.times_s, want.times_s(:)', 'AbsTol', 1e-6, profile);
                tc.verifyEqual(st.channel_x, want.channel_x(:)', profile);
                tc.verifyEqual(st.channel_y, want.channel_y(:)', profile);
                tc.verifyEqual(st.pattern_a, want.pattern_a(:)', profile);
                tc.verifyEqual(st.pattern_b, want.pattern_b(:)', profile);
                tc.verifyEqual(st.noise_channel, want.noise_channel, profile);
                tc.verifyEqual(st.nan_channels, want.nan_channels, profile);
                tc.verifyEqual(st.onset_index, want.onset_index, profile);
            end
        end

        function datasetHasTheDocumentedShapeAndMetadata(tc)
            [ds, truth] = prismt.demo.makeSyntheticDataset(Profile="fast");
            tc.verifyEqual([ds.N ds.R ds.T ds.M], [480 12 10 2]);
            tc.verifyEqual(string(ds.Trials.Properties.VariableNames), ["mouse", "session", "phase", "stim", "response"]);
            tc.verifyEqual(ds.Subject, "mouse");
            tc.verifyEqual(numel(unique(ds.Trials.mouse)), 8);
            tc.verifyEqual(truth.StimulusChannels, [9 10]);
            tc.verifyEmpty(ds.validate());
        end

        function plantedMissingChannelsAreNaN(tc)
            [ds, truth] = prismt.demo.makeSyntheticDataset(Profile="tiny");
            for k = 1:size(truth.MissingChannels, 1)
                mouse = categorical(compose("M%02d", truth.MissingChannels(k, 1)));
                tc.verifyTrue(all(isnan(ds.X(ds.Trials.mouse == mouse, truth.MissingChannels(k, 2), :, :)), 'all'));
            end
        end

        function stimIsBalancedWithinSessions(tc)
            ds = prismt.demo.makeSyntheticDataset(Profile="fast", Seed=4);
            g = findgroups(ds.Trials.mouse, ds.Trials.session);
            counts = splitapply(@sum, ds.Trials.stim, g);
            tc.verifyEqual(counts, repmat(15, size(counts)));
        end

        function plantedResponsesAreVisibleInTheAverages(tc)
            [ds, truth] = prismt.demo.makeSyntheticDataset(Profile="fast", Difficulty="easy");
            post = truth.OnsetIndex:ds.T;
            a = truth.StimulusChannels;
            b = truth.LearningChannels;
            cs = mean(ds.X(ds.Trials.stim == 1, a, post, 1), 'all', 'omitnan') - ...
                 mean(ds.X(ds.Trials.stim == 0, a, post, 1), 'all', 'omitnan');
            learn = mean(ds.X(ds.Trials.phase == "late", b, post, 2), 'all', 'omitnan') - ...
                    mean(ds.X(ds.Trials.phase == "early", b, post, 2), 'all', 'omitnan');
            tc.verifyGreaterThan(cs, 1.0);
            tc.verifyGreaterThan(learn, 1.0);
        end

        function seedMakesItReproducibleAndLeavesGlobalRngAlone(tc)
            rng(123);
            before = rand;
            rng(123);
            a = prismt.demo.makeSyntheticDataset(Profile="tiny", Seed=7);
            after = rand;
            b = prismt.demo.makeSyntheticDataset(Profile="tiny", Seed=7);
            tc.verifyEqual(before, after);
            tc.verifyEqual(a.X, b.X);
        end
    end
end
