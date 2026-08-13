%% phase3_step2_mlp.m
% Phase 3, Step 2: MLP (feedforward neural network) baseline, substituting
% for the unavailable XGBoost/boosted-trees baseline (Statistics and
% Machine Learning Toolbox not available on this license). Uses Deep
% Learning Toolbox's fitnet.
%
% Features per row: lags 1-4 of Power_kW, plus whatever weather columns
% are NOT entirely NaN for that dataset (Irradiance_Wm2, Temp_C,
% Humidity_pct, Wind_ms). Target: Power_kW at time t.
% One pooled MLP trained per DATASET (all sites' TRAIN rows together),
% evaluated per-site on that site's TEST rows -- consistent with how the
% Persistence baseline (Step 1) was evaluated, so results are comparable.

clear; clc;

baseDir = 'C:\Users\Shaho\Desktop\claude_projects\R9\pv_forecasting_enchmark\dataset';
protocolDir = fullfile(baseDir, 'protocol');
outDir = fullfile(baseDir, 'results');
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = {
    'dataset1_DKASC_15min_labeled.csv'
    'dataset2_HKUST_15min_labeled.csv'
    'dataset3_Ausgrid_raw_harmonized_labeled.csv'
    'dataset4_PVDAQ_15min_labeled.csv'
};

nLags = 4;
weatherCandidates = {'Irradiance_Wm2','Temp_C','Humidity_pct','Wind_ms'};

allResults = table();

for f = 1:numel(files)
    fPath = fullfile(protocolDir, files{f});
    if ~isfile(fPath)
        warning('File not found, skipping: %s', fPath);
        continue
    end
    fprintf('--- MLP baseline: %s ---\n', files{f});

    opts = detectImportOptions(fPath, 'VariableNamingRule', 'preserve');
    opts = setvartype(opts, 'Site_ID', 'string');
    T = readtable(fPath, opts);
    T.Timestamp = datetime(T.Timestamp);
    if iscell(T.Split); T.Split = string(T.Split); end
    if ~islogical(T.IsRareEvent)
        T.IsRareEvent = logical(T.IsRareEvent);
    end

    % Determine which weather columns are usable (not entirely NaN) for
    % this dataset -- Ausgrid has none, PVDAQ subset only has Irradiance.
    usableWeather = {};
    for w = 1:numel(weatherCandidates)
        if ismember(weatherCandidates{w}, T.Properties.VariableNames) && any(~isnan(T.(weatherCandidates{w})))
            usableWeather{end+1} = weatherCandidates{w}; %#ok<AGROW>
        end
    end
    fprintf('  Usable weather features: %s\n', strjoin(usableWeather, ', '));

    % --- Build lag-feature table, per site (lags must not cross site boundaries) ---
    sites = unique(T.Site_ID);
    featureRows = {};
    for i = 1:numel(sites)
        Tsite = T(T.Site_ID == sites(i), :);
        Tsite = sortrows(Tsite, 'Timestamp');
        n = height(Tsite);
        if n <= nLags
            continue
        end
        p = Tsite.Power_kW;
        lagMat = NaN(n, nLags);
        for L = 1:nLags
            lagMat(L+1:end, L) = p(1:end-L);
        end
        Wmat = [];
        if ~isempty(usableWeather)
            Wmat = Tsite{:, usableWeather};
            % Impute isolated gaps (forward-fill, then mean-fill any
            % still-missing leading values) instead of letting a single
            % missing weather reading wipe out an otherwise-valid row --
            % this was silently destroying most of DKASC's data since its
            % 4 weather columns rarely have simultaneous coverage.
            for c = 1:size(Wmat, 2)
                col = Wmat(:, c);
                col = fillmissing(col, 'previous');
                col = fillmissing(col, 'constant', mean(col, 'omitnan'));
                Wmat(:, c) = col;
            end
        end

        % Cyclical time features (always available, unlike weather --
        % critical for Ausgrid which has NO weather data at all, meaning
        % without this the model has zero sun-angle/time-of-day signal).
        hourFrac = hour(Tsite.Timestamp) + minute(Tsite.Timestamp)/60;
        doy = day(Tsite.Timestamp, 'dayofyear');
        timeFeat = [sin(2*pi*hourFrac/24), cos(2*pi*hourFrac/24), ...
                    sin(2*pi*doy/365.25), cos(2*pi*doy/365.25)];
        Wmat = [Wmat, timeFeat];
        % Normalize by this site's OWN capacity (P99.5 of its own observed
        % power) before pooling into the shared MLP -- without this,
        % pooling raw kW across sites spanning 6kW to 408kW makes the
        % network's learned scale meaningless for most sites (confirmed:
        % this was the cause of PVDAQ's 479% MAPE and HKUST underperforming
        % persistence in the first run).
        siteCap = prctile(p(p > 0), 99.5);
        if isnan(siteCap) || siteCap <= 0
            siteCap = max(p, [], 'omitnan');
        end
        if isnan(siteCap) || siteCap <= 0
            continue % genuinely no usable signal for this site
        end
        lagMat = lagMat / siteCap;
        p_norm = p / siteCap;

        Xi = [lagMat, Wmat];
        yi = p_norm;
        splitI = Tsite.Split;
        siteIDi = Tsite.Site_ID;
        capI = repmat(siteCap, n, 1);
        rareI = Tsite.IsRareEvent;
        featureRows{end+1} = struct('X', Xi, 'y', yi, 'Split', splitI, 'Site_ID', siteIDi, 'Cap', capI, 'Rare', rareI); %#ok<AGROW>
    end

    % Concatenate all sites
    Xall = []; yall = []; splitAll = strings(0,1); siteAll = strings(0,1); capAll = []; rareAll = logical.empty(0,1);
    for i = 1:numel(featureRows)
        r = featureRows{i};
        Xall = [Xall; r.X]; %#ok<AGROW>
        yall = [yall; r.y]; %#ok<AGROW>
        splitAll = [splitAll; r.Split]; %#ok<AGROW>
        siteAll = [siteAll; r.Site_ID]; %#ok<AGROW>
        capAll = [capAll; r.Cap]; %#ok<AGROW>
        rareAll = [rareAll; r.Rare]; %#ok<AGROW>
    end

    validRow = all(~isnan(Xall), 2) & ~isnan(yall);
    Xall = Xall(validRow, :); yall = yall(validRow); splitAll = splitAll(validRow); siteAll = siteAll(validRow); capAll = capAll(validRow); rareAll = rareAll(validRow);

    trainIdx = splitAll == "train";
    testIdx  = splitAll == "test";

    if sum(trainIdx) < 50 || sum(testIdx) < 10
        warning('  Not enough valid rows to train/test MLP for %s -- skipping.', files{f});
        continue
    end

    % --- Train pooled MLP on this dataset's TRAIN rows ---
    % fitnet's default algorithm (Levenberg-Marquardt) does not scale to
    % datasets with hundreds of thousands to millions of rows (this is
    % why HKUST/Ausgrid/PVDAQ would hang indefinitely). Fix: switch to
    % scaled conjugate gradient (trainscg, no Hessian approximation, scales
    % to large N), and cap the pooled training set via random subsampling
    % so training completes in reasonable time even for Ausgrid's 15.7M rows.
    net = fitnet([16 8], 'trainscg');
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = 0.85; % fitnet's own internal early-stopping split
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0;

    maxTrainRows = 150000;
    trainRowIdx = find(trainIdx);
    if numel(trainRowIdx) > maxTrainRows
        rng(42); % reproducible subsample
        trainRowIdx = trainRowIdx(randperm(numel(trainRowIdx), maxTrainRows));
    end

    Xtrain = Xall(trainRowIdx, :)';
    ytrain = yall(trainRowIdx)';

    fprintf('  Training MLP on %d rows (subsampled from %d available)...\n', numel(trainRowIdx), sum(trainIdx));
    net = train(net, Xtrain, ytrain);

    % --- Evaluate per site on TEST rows ---
    for i = 1:numel(sites)
        siteMask = testIdx & (siteAll == sites(i));
        if sum(siteMask) < 2
            continue
        end
        Xtest = Xall(siteMask, :)';
        ytest_norm = yall(siteMask);
        ypred_norm = net(Xtest)';

        % De-normalize back to real kW using this site's own capacity,
        % so RMSE/MAE/MAPE are reported in the same real-unit terms as
        % the Persistence baseline (fair, direct comparison).
        siteCapEval = capAll(siteMask);
        ytest = ytest_norm .* siteCapEval;
        ypred = ypred_norm .* siteCapEval;

        [rmse, mae, mape] = computeMetrics(ytest, ypred);

        rareMaskEval = logical(rareAll(siteMask));
        if sum(rareMaskEval) >= 2
            [rmseRare, maeRare, mapeRare] = computeMetrics(ytest(rareMaskEval), ypred(rareMaskEval));
        else
            rmseRare = NaN; maeRare = NaN; mapeRare = NaN;
        end

        newRow = table(string(files{f}), sites(i), "MLP", sum(siteMask), rmse, mae, mape, ...
            sum(rareMaskEval), rmseRare, maeRare, mapeRare, ...
            'VariableNames', {'Dataset','Site_ID','Model','N_test','RMSE','MAE','MAPE', ...
            'N_RareEvent','RMSE_RareEvent','MAE_RareEvent','MAPE_RareEvent'});
        allResults = [allResults; newRow]; %#ok<AGROW>
    end
end

allResults.Site_ID = "ID_" + allResults.Site_ID;
writetable(allResults, fullfile(outDir, 'phase3_mlp_results.csv'));

fprintf('\n=== MLP baseline summary (mean across sites) ===\n');
datasets = unique(allResults.Dataset);
for d = 1:numel(datasets)
    sub = allResults(allResults.Dataset == datasets(d), :);
    fprintf('%s: RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  (n_sites=%d)\n', ...
        datasets(d), mean(sub.RMSE, 'omitnan'), mean(sub.MAE, 'omitnan'), ...
        mean(sub.MAPE, 'omitnan'), height(sub));
end

fprintf('\nResults written to: %s\n', fullfile(outDir, 'phase3_mlp_results.csv'));

function [rmse, mae, mape] = computeMetrics(yTrue, yPred)
    err = yTrue - yPred;
    rmse = sqrt(mean(err.^2, 'omitnan'));
    mae = mean(abs(err), 'omitnan');
    nonZero = abs(yTrue) > 0.01;
    if any(nonZero)
        mape = 100 * mean(abs(err(nonZero) ./ yTrue(nonZero)));
    else
        mape = NaN;
    end
end
