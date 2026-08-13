%% phase4_leaderboard.m
% Phase 4: consolidates all Phase 3 baseline results (Persistence, MLP,
% LSTM, Transformer) into one master leaderboard table -- the core
% "benchmark results table" deliverable from draft.md Section 7, and the
% cross-climate degradation summary that is the paper's central novel
% result (draft.md Research Question).

clear; clc;

baseDir = 'C:\Users\Shaho\Desktop\claude_projects\R9\pv_forecasting_enchmark\dataset';
resultsDir = fullfile(baseDir, 'results');
outDir = fullfile(baseDir, 'leaderboard');
if ~exist(outDir, 'dir'); mkdir(outDir); end

resultFiles = {
    'phase3_persistence_results.csv'
    'phase3_mlp_results.csv'
    'phase3_lstm_results.csv'
    'phase3_transformer_results.csv'
};

allResults = table();
for r = 1:numel(resultFiles)
    fPath = fullfile(resultsDir, resultFiles{r});
    if ~isfile(fPath)
        warning('Missing results file: %s', fPath);
        continue
    end
    opts = detectImportOptions(fPath, 'VariableNamingRule', 'preserve');
    opts = setvartype(opts, {'Dataset','Site_ID','Model'}, 'string');
    Tr = readtable(fPath, opts);
    allResults = [allResults; Tr]; %#ok<AGROW>
end

% Clean dataset names for readability
allResults.DatasetShort = extractBefore(allResults.Dataset, "_labeled.csv");
allResults.DatasetShort = strrep(allResults.DatasetShort, "_15min", "");
allResults.DatasetShort = strrep(allResults.DatasetShort, "_raw_harmonized", "");

%% ---- Master leaderboard: mean metrics per Dataset x Model ----
models = unique(allResults.Model);
datasets = unique(allResults.DatasetShort);

leaderboard = table();
for d = 1:numel(datasets)
    for m = 1:numel(models)
        sub = allResults(allResults.DatasetShort == datasets(d) & allResults.Model == models(m), :);
        if isempty(sub)
            continue
        end
        newRow = table(datasets(d), models(m), height(sub), ...
            mean(sub.RMSE, 'omitnan'), mean(sub.MAE, 'omitnan'), mean(sub.MAPE, 'omitnan'), ...
            'VariableNames', {'Dataset','Model','N_sites','Mean_RMSE','Mean_MAE','Mean_MAPE'});
        leaderboard = [leaderboard; newRow]; %#ok<AGROW>
    end
end

leaderboard = sortrows(leaderboard, {'Dataset','Mean_RMSE'});
writetable(leaderboard, fullfile(outDir, 'SolarBench_leaderboard.csv'));

fprintf('=== SolarBench Leaderboard (mean across sites, lower is better) ===\n\n');
for d = 1:numel(datasets)
    fprintf('--- %s ---\n', datasets(d));
    sub = leaderboard(leaderboard.Dataset == datasets(d), :);
    sub = sortrows(sub, 'Mean_RMSE');
    for i = 1:height(sub)
        fprintf('  %-12s RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  (n_sites=%d)\n', ...
            sub.Model(i), sub.Mean_RMSE(i), sub.Mean_MAE(i), sub.Mean_MAPE(i), sub.N_sites(i));
    end
    [~, bestIdx] = min(sub.Mean_RMSE);
    fprintf('  --> Best by RMSE: %s\n\n', sub.Model(bestIdx));
end

%% ---- Cross-climate degradation summary ----
% For each model, how much does its RMSE vary across the 4 climate-
% distinct datasets? Large spread = poor cross-climate generalization,
% directly answering the core SolarBench research question.
fprintf('=== Cross-climate RMSE spread per model (the core novel result) ===\n\n');
crossClimate = table();
for m = 1:numel(models)
    sub = leaderboard(leaderboard.Model == models(m), :);
    if height(sub) < 2
        continue
    end
    rmseRange = max(sub.Mean_RMSE) - min(sub.Mean_RMSE);
    rmseRatio = max(sub.Mean_RMSE) / max(min(sub.Mean_RMSE), eps);
    fprintf('  %-12s RMSE range across datasets: %.4f (min=%.4f, max=%.4f, ratio=%.1fx)\n', ...
        models(m), rmseRange, min(sub.Mean_RMSE), max(sub.Mean_RMSE), rmseRatio);
    newRow = table(models(m), min(sub.Mean_RMSE), max(sub.Mean_RMSE), rmseRange, rmseRatio, ...
        'VariableNames', {'Model','Min_RMSE','Max_RMSE','RMSE_Range','RMSE_Ratio'});
    crossClimate = [crossClimate; newRow]; %#ok<AGROW>
end
writetable(crossClimate, fullfile(outDir, 'SolarBench_cross_climate_degradation.csv'));

%% ---- Rare-event performance (the meta-gap this whole benchmark targets) ----
% Literature repeatedly flags that models are evaluated only on normal
% conditions, with rare/extreme events (clipping, cloud transients)
% structurally under-tested. This directly reports that gap.
fprintf('\n=== Rare-event vs overall performance (RMSE degradation) ===\n\n');
rareLeaderboard = table();
for d = 1:numel(datasets)
    for m = 1:numel(models)
        sub = allResults(allResults.DatasetShort == datasets(d) & allResults.Model == models(m), :);
        if isempty(sub) || all(isnan(sub.RMSE_RareEvent))
            continue
        end
        overallRMSE = mean(sub.RMSE, 'omitnan');
        rareRMSE = mean(sub.RMSE_RareEvent, 'omitnan');
        degradePct = 100 * (rareRMSE - overallRMSE) / overallRMSE;
        fprintf('  %-12s %-15s Overall RMSE=%.4f  RareEvent RMSE=%.4f  (%.1f%% worse on rare events)\n', ...
            datasets(d), models(m), overallRMSE, rareRMSE, degradePct);
        newRow = table(datasets(d), models(m), overallRMSE, rareRMSE, degradePct, ...
            'VariableNames', {'Dataset','Model','Overall_RMSE','RareEvent_RMSE','PctWorse_RareEvent'});
        rareLeaderboard = [rareLeaderboard; newRow]; %#ok<AGROW>
    end
end
writetable(rareLeaderboard, fullfile(outDir, 'SolarBench_rare_event_performance.csv'));

fprintf('\nLeaderboard written to: %s\n', fullfile(outDir, 'SolarBench_leaderboard.csv'));
fprintf('Cross-climate degradation summary written to: %s\n', fullfile(outDir, 'SolarBench_cross_climate_degradation.csv'));
