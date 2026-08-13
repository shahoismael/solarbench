%% phase3_step3_lstm.m
% Phase 3, Step 3: LSTM baseline (Deep Learning Toolbox).
%
% Same per-site capacity normalization lesson learned from the MLP step
% (Step 2) applied here from the start. Sequence length = 8 steps (2h at
% 15-min resolution, or 4h at 30-min for Ausgrid) of [Power_norm, usable
% weather features], predicting the next single Power_norm value.
% Sequences never cross site boundaries. Pooled training per dataset,
% subsampled to a max count for tractable runtime (same cap philosophy as
% Step 2), evaluated per-site on TEST rows, de-normalized back to kW.

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

seqLen = 8;
weatherCandidates = {'Irradiance_Wm2','Temp_C','Humidity_pct','Wind_ms'};
maxTrainSeqs = 100000; % smaller cap than MLP -- LSTM sequences cost much more per-sample

allResults = table();

for f = 1:numel(files)
    fPath = fullfile(protocolDir, files{f});
    if ~isfile(fPath)
        warning('File not found, skipping: %s', fPath);
        continue
    end
    fprintf('--- LSTM baseline: %s ---\n', files{f});

    opts = detectImportOptions(fPath, 'VariableNamingRule', 'preserve');
    opts = setvartype(opts, 'Site_ID', 'string');
    T = readtable(fPath, opts);
    T.Timestamp = datetime(T.Timestamp);
    if iscell(T.Split); T.Split = string(T.Split); end
    if ~islogical(T.IsRareEvent)
        T.IsRareEvent = logical(T.IsRareEvent);
    end

    usableWeather = {};
    for w = 1:numel(weatherCandidates)
        if ismember(weatherCandidates{w}, T.Properties.VariableNames) && any(~isnan(T.(weatherCandidates{w})))
            usableWeather{end+1} = weatherCandidates{w}; %#ok<AGROW>
        end
    end
    fprintf('  Usable weather features: %s\n', strjoin(usableWeather, ', '));
    nFeat = 1 + numel(usableWeather) + 4; % +4 for cyclical hour/day-of-year sin-cos features

    sites = unique(T.Site_ID);

    % --- Build sliding-window sequences per site (never crossing site boundaries) ---
    Xtrain_seq = {}; ytrain_seq = [];
    Xtest_seq  = {}; ytest_seq  = []; testSite = strings(0,1); testCap = []; testRare = logical.empty(0,1);

    for i = 1:numel(sites)
        Tsite = T(T.Site_ID == sites(i), :);
        Tsite = sortrows(Tsite, 'Timestamp');
        n = height(Tsite);
        if n <= seqLen + 1
            continue
        end
        p = Tsite.Power_kW;
        siteCap = prctile(p(p > 0), 99.5);
        if isnan(siteCap) || siteCap <= 0
            siteCap = max(p, [], 'omitnan');
        end
        if isnan(siteCap) || siteCap <= 0
            continue
        end
        p_norm = p / siteCap;

        Wmat = [];
        if ~isempty(usableWeather)
            Wmat = Tsite{:, usableWeather};
            for c = 1:size(Wmat, 2)
                col = fillmissing(Wmat(:,c), 'previous');
                col = fillmissing(col, 'constant', mean(col, 'omitnan'));
                Wmat(:, c) = col;
            end
        end

        % Cyclical time features (always available, unlike weather --
        % critical for Ausgrid which has NO weather data at all).
        hourFrac = hour(Tsite.Timestamp) + minute(Tsite.Timestamp)/60;
        doy = day(Tsite.Timestamp, 'dayofyear');
        timeFeat = [sin(2*pi*hourFrac/24), cos(2*pi*hourFrac/24), ...
                    sin(2*pi*doy/365.25), cos(2*pi*doy/365.25)];
        Wmat = [Wmat, timeFeat];

        Fmat = [p_norm, Wmat]'; % nFeat x n (feature-major, as trainnet sequence format expects)

        splitVec = Tsite.Split;
        rareVec = Tsite.IsRareEvent;

        for t = (seqLen+1):n
            seq = Fmat(:, t-seqLen:t-1)'; % transposed to [seqLen x nFeat] -- trainnet
                                          % expects time-steps-first orientation, unlike
                                          % legacy trainNetwork's [nFeat x seqLen]
            target = p_norm(t);
            if any(isnan(seq(:))) || isnan(target)
                continue
            end
            if splitVec(t) == "train"
                Xtrain_seq{end+1} = seq; %#ok<AGROW>
                ytrain_seq(end+1) = target; %#ok<AGROW>
            elseif splitVec(t) == "test"
                Xtest_seq{end+1} = seq; %#ok<AGROW>
                ytest_seq(end+1) = target; %#ok<AGROW>
                testSite(end+1) = sites(i); %#ok<AGROW>
                testCap(end+1) = siteCap; %#ok<AGROW>
                testRare(end+1) = rareVec(t); %#ok<AGROW>
            end
        end
    end

    if numel(Xtrain_seq) < 100 || numel(Xtest_seq) < 10
        warning('  Not enough sequences for %s -- skipping.', files{f});
        continue
    end

    if numel(Xtrain_seq) > maxTrainSeqs
        rng(42);
        keepIdx = randperm(numel(Xtrain_seq), maxTrainSeqs);
        Xtrain_seq = Xtrain_seq(keepIdx);
        ytrain_seq = ytrain_seq(keepIdx);
    end

    fprintf('  Training LSTM on %d sequences (test pool: %d)...\n', numel(Xtrain_seq), numel(Xtest_seq));

    layers = [
        sequenceInputLayer(nFeat)
        lstmLayer(32, 'OutputMode', 'last')
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(1)
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 256, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 200);

    net = trainnet(Xtrain_seq', ytrain_seq', layers, 'mse', options);

    % --- Evaluate per site on TEST sequences ---
    ypred_all = minibatchpredict(net, Xtest_seq');
    ypred_all = double(ypred_all);

    for i = 1:numel(sites)
        siteMask = testSite == sites(i);
        if sum(siteMask) < 2
            continue
        end
        cap = testCap(find(siteMask, 1));
        yt = ytest_seq(siteMask)' .* cap;
        yp = ypred_all(siteMask) .* cap;

        [rmse, mae, mape] = computeMetrics(yt, yp);

        rareMaskEval = logical(testRare(siteMask));
        if sum(rareMaskEval) >= 2
            [rmseRare, maeRare, mapeRare] = computeMetrics(yt(rareMaskEval), yp(rareMaskEval));
        else
            rmseRare = NaN; maeRare = NaN; mapeRare = NaN;
        end

        newRow = table(string(files{f}), sites(i), "LSTM", sum(siteMask), rmse, mae, mape, ...
            sum(rareMaskEval), rmseRare, maeRare, mapeRare, ...
            'VariableNames', {'Dataset','Site_ID','Model','N_test','RMSE','MAE','MAPE', ...
            'N_RareEvent','RMSE_RareEvent','MAE_RareEvent','MAPE_RareEvent'});
        allResults = [allResults; newRow]; %#ok<AGROW>
    end
end

allResults.Site_ID = "ID_" + allResults.Site_ID;
writetable(allResults, fullfile(outDir, 'phase3_lstm_results.csv'));

fprintf('\n=== LSTM baseline summary (mean across sites) ===\n');
datasets = unique(allResults.Dataset);
for d = 1:numel(datasets)
    sub = allResults(allResults.Dataset == datasets(d), :);
    fprintf('%s: RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  (n_sites=%d)\n', ...
        datasets(d), mean(sub.RMSE, 'omitnan'), mean(sub.MAE, 'omitnan'), ...
        mean(sub.MAPE, 'omitnan'), height(sub));
end

fprintf('\nResults written to: %s\n', fullfile(outDir, 'phase3_lstm_results.csv'));

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
