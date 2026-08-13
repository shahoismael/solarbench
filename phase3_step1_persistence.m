%% phase3_step1_persistence.m
% Phase 3, Step 1: Persistence baseline (naive forecast = previous
% timestep's value) + the shared evaluation harness (RMSE/MAE/MAPE) that
% all later models (MLP, LSTM, Transformer) will reuse.
%
% Reads the Phase 2 labeled/split files from the "protocol" folder.
% Evaluates ONLY on each site's own "test" split, per-site, then reports
% both per-site and dataset-wide aggregate metrics.
%
% NOTE ON BASELINE SET: draft.md originally specified an XGBoost/boosted-
% trees baseline via fitrensemble (Statistics and Machine Learning
% Toolbox), which is not available on this machine/license. Substituted
% baseline set for Phase 3: Persistence, MLP (feedforward NN, Deep
% Learning Toolbox), LSTM, Transformer -- all buildable with the
% confirmed available toolboxes.

clear; clc;

baseDir = solarbench_config();   % resolves SOLARBENCH_DATA, else <repo>/data
protocolDir = fullfile(baseDir, 'protocol');
outDir = fullfile(baseDir, 'results');
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = {
    'dataset1_DKASC_15min_labeled.csv'
    'dataset2_HKUST_15min_labeled.csv'
    'dataset3_Ausgrid_raw_harmonized_labeled.csv'
    'dataset4_PVDAQ_15min_labeled.csv'
};

allResults = table();

for f = 1:numel(files)
    fPath = fullfile(protocolDir, files{f});
    if ~isfile(fPath)
        warning('File not found, skipping: %s', fPath);
        continue
    end
    fprintf('--- Persistence baseline: %s ---\n', files{f});
    opts = detectImportOptions(fPath, 'VariableNamingRule', 'preserve');
    opts = setvartype(opts, 'Site_ID', 'string');
    T = readtable(fPath, opts);
    T.Timestamp = datetime(T.Timestamp);
    if iscell(T.Split); T.Split = string(T.Split); end
    if ~islogical(T.IsRareEvent)
        T.IsRareEvent = logical(T.IsRareEvent);
    end

    sites = unique(T.Site_ID);
    for i = 1:numel(sites)
        Tsite = T(T.Site_ID == sites(i), :);
        Tsite = sortrows(Tsite, 'Timestamp');

        % Persistence prediction: y_hat(t) = y(t-1), evaluated only on
        % rows whose split label is "test" AND whose previous row exists
        % and is not itself a gap (checked via time-step consistency).
        y = Tsite.Power_kW;
        yhat = [NaN; y(1:end-1)];

        isTest = Tsite.Split == "test";
        validRows = isTest & ~isnan(y) & ~isnan(yhat);

        if sum(validRows) < 2
            continue
        end

        [rmse, mae, mape] = computeMetrics(y(validRows), yhat(validRows));

        % Rare-event-specific metrics -- the whole point of Phase 2's
        % IsRareEvent labeling was to evaluate performance on rare/extreme
        % events separately, which was built but never actually reported.
        rareRows = validRows & Tsite.IsRareEvent;
        if sum(rareRows) >= 2
            [rmseRare, maeRare, mapeRare] = computeMetrics(y(rareRows), yhat(rareRows));
        else
            rmseRare = NaN; maeRare = NaN; mapeRare = NaN;
        end

        newRow = table(string(files{f}), sites(i), "Persistence", sum(validRows), rmse, mae, mape, ...
            sum(rareRows), rmseRare, maeRare, mapeRare, ...
            'VariableNames', {'Dataset','Site_ID','Model','N_test','RMSE','MAE','MAPE', ...
            'N_RareEvent','RMSE_RareEvent','MAE_RareEvent','MAPE_RareEvent'});
        allResults = [allResults; newRow]; %#ok<AGROW>
    end
end

allResults.Site_ID = "ID_" + allResults.Site_ID; % guarantee text-type on re-read, since
                                                    % Ausgrid/PVDAQ site names are pure
                                                    % numbers and get combined here with
                                                    % HKUST/DKASC's alphabetic names in one file
writetable(allResults, fullfile(outDir, 'phase3_persistence_results.csv'));

% Aggregate summary per dataset
fprintf('\n=== Persistence baseline summary (mean across sites) ===\n');
datasets = unique(allResults.Dataset);
for d = 1:numel(datasets)
    sub = allResults(allResults.Dataset == datasets(d), :);
    fprintf('%s: RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  (n_sites=%d)\n', ...
        datasets(d), mean(sub.RMSE, 'omitnan'), mean(sub.MAE, 'omitnan'), ...
        mean(sub.MAPE, 'omitnan'), height(sub));
end

fprintf('\nResults written to: %s\n', fullfile(outDir, 'phase3_persistence_results.csv'));

%% ---- Shared evaluation harness (reused by all later Phase 3 models) ----
function [rmse, mae, mape] = computeMetrics(yTrue, yPred)
    err = yTrue - yPred;
    rmse = sqrt(mean(err.^2, 'omitnan'));
    mae = mean(abs(err), 'omitnan');
    % MAPE computed only over rows where true value is meaningfully
    % non-zero, to avoid division-by-near-zero blowups at night.
    nonZero = abs(yTrue) > 0.01;
    if any(nonZero)
        mape = 100 * mean(abs(err(nonZero) ./ yTrue(nonZero)));
    else
        mape = NaN;
    end
end
