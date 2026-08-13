%% phase3_step4_transformer.m
% Phase 3, Step 4: Transformer baseline (Deep Learning Toolbox).
%
% Same data pipeline as Step 3 (LSTM): per-site capacity normalization,
% sliding-window sequences (seqLen=8), pooled training per dataset,
% subsampled for tractable runtime, evaluated per-site on TEST, de-
% normalized to real kW. Only the architecture changes: a small
% Transformer encoder (self-attention) instead of an LSTM layer.

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

seqLen = 8;
weatherCandidates = {'Irradiance_Wm2','Temp_C','Humidity_pct','Wind_ms'};
maxTrainSeqs = 100000;

allResults = table();

for f = 1:numel(files)
    fPath = fullfile(protocolDir, files{f});
    if ~isfile(fPath)
        warning('File not found, skipping: %s', fPath);
        continue
    end
    fprintf('--- Transformer baseline: %s ---\n', files{f});

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

        Fmat = [p_norm, Wmat]'; % nFeat x n

        splitVec = Tsite.Split;
        rareVec = Tsite.IsRareEvent;

        for t = (seqLen+1):n
            seq = Fmat(:, t-seqLen:t-1)'; % [seqLen x nFeat] -- matches trainnet convention confirmed in Step 3
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

    fprintf('  Training Transformer on %d sequences (test pool: %d)...\n', numel(Xtrain_seq), numel(Xtest_seq));

    embedDim = 32; % must be divisible by numHeads
    numHeads = 4;

    % Built as an explicit layerGraph (not a plain layer array). This is
    % essential: in a sequential layer array, positionEmbeddingLayer's
    % output would REPLACE the feature-embedding output rather than being
    % ADDED to it (plain arrays have no skip/residual connections), which
    % meant attention was seeing pure position information with the
    % actual power/weather signal completely discarded -- the real cause
    % of the training-loss plateau in every previous attempt.
    lgraph = layerGraph();
    lgraph = addLayers(lgraph, sequenceInputLayer(nFeat, 'Name', 'input'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(embedDim, 'Name', 'embed'));
    lgraph = addLayers(lgraph, positionEmbeddingLayer(embedDim, seqLen, 'Name', 'posembed'));
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'add1'));
    lgraph = addLayers(lgraph, selfAttentionLayer(numHeads, embedDim, 'Name', 'attn'));
    lgraph = addLayers(lgraph, layerNormalizationLayer('Name', 'norm1'));
    lgraph = addLayers(lgraph, lstmLayer(embedDim, 'OutputMode', 'last', 'Name', 'readout'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(16, 'Name', 'fc1'));
    lgraph = addLayers(lgraph, reluLayer('Name', 'relu2'));
    lgraph = addLayers(lgraph, fullyConnectedLayer(1, 'Name', 'fc_out'));

    lgraph = connectLayers(lgraph, 'input', 'embed');
    lgraph = connectLayers(lgraph, 'embed', 'posembed');
    lgraph = connectLayers(lgraph, 'embed', 'add1/in1');
    lgraph = connectLayers(lgraph, 'posembed', 'add1/in2');
    lgraph = connectLayers(lgraph, 'add1', 'attn');
    lgraph = connectLayers(lgraph, 'attn', 'norm1');
    lgraph = connectLayers(lgraph, 'norm1', 'readout');
    lgraph = connectLayers(lgraph, 'readout', 'fc1');
    lgraph = connectLayers(lgraph, 'fc1', 'relu2');
    lgraph = connectLayers(lgraph, 'relu2', 'fc_out');

    layers = dlnetwork(lgraph);

    % Adam's default learning rate (0.001) is well-documented as too
    % aggressive for attention models without a warmup schedule -- lowered
    % here to 1e-4, which was likely compounding the bug above.
    options = trainingOptions('adam', ...
        'MaxEpochs', 15, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 200);

    net = trainnet(Xtrain_seq', ytrain_seq', layers, 'mse', options);

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

        newRow = table(string(files{f}), sites(i), "Transformer", sum(siteMask), rmse, mae, mape, ...
            sum(rareMaskEval), rmseRare, maeRare, mapeRare, ...
            'VariableNames', {'Dataset','Site_ID','Model','N_test','RMSE','MAE','MAPE', ...
            'N_RareEvent','RMSE_RareEvent','MAE_RareEvent','MAPE_RareEvent'});
        allResults = [allResults; newRow]; %#ok<AGROW>
    end
end

allResults.Site_ID = "ID_" + allResults.Site_ID;
writetable(allResults, fullfile(outDir, 'phase3_transformer_results.csv'));

fprintf('\n=== Transformer baseline summary (mean across sites) ===\n');
datasets = unique(allResults.Dataset);
for d = 1:numel(datasets)
    sub = allResults(allResults.Dataset == datasets(d), :);
    fprintf('%s: RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  (n_sites=%d)\n', ...
        datasets(d), mean(sub.RMSE, 'omitnan'), mean(sub.MAE, 'omitnan'), ...
        mean(sub.MAPE, 'omitnan'), height(sub));
end

fprintf('\nResults written to: %s\n', fullfile(outDir, 'phase3_transformer_results.csv'));

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
