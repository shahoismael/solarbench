%% phase2_protocol_design.m
% Phase 2 driver: for each harmonized dataset, builds fixed chronological
% train/val/test splits (NO random shuffling -- time-series leakage risk)
% and a labeled rare-event subset (clipping, low-irradiance/cloud
% transients). Reads from the "harmonized" folder produced by
% run_harmonization.m (Phase 1) and writes split-label columns back out.
%
% Split ratio: 70% train / 15% val / 15% test, applied chronologically
% PER SITE (so each site's own timeline is split independently, avoiding
% one site's early data leaking into another site's "test" period).

clear; clc;

baseDir = 'C:\Users\Shaho\Desktop\claude_projects\R9\pv_forecasting_enchmark\dataset';
harmDir = fullfile(baseDir, 'harmonized');
outDir  = fullfile(baseDir, 'protocol');
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = {
    'dataset1_DKASC_15min.csv'
    'dataset2_HKUST_15min.csv'
    'dataset3_Ausgrid_raw_harmonized.csv'  % native half-hourly, not resampled
    'dataset4_PVDAQ_15min.csv'
};

for f = 1:numel(files)
    fPath = fullfile(harmDir, files{f});
    if ~isfile(fPath)
        warning('File not found, skipping: %s', fPath);
        continue
    end
    fprintf('--- Processing %s ---\n', files{f});
    T = readtable(fPath, 'VariableNamingRule', 'preserve');
    T.Timestamp = datetime(T.Timestamp);
    if iscell(T.Site_ID)
        T.Site_ID = string(T.Site_ID);
    end

    % Split strategy: GLOBAL calendar-date split works when all sites in
    % a dataset share one installation timeline (DKASC, HKUST, Ausgrid).
    % PVDAQ's 8 sites were deliberately selected for climate diversity,
    % NOT shared timing, and span installation dates from 2007 to 2020+
    % -- a single global date window would exclude most of them. So
    % PVDAQ specifically uses a per-site percentage split instead.
    if contains(files{f}, 'PVDAQ')
        T = addSplitLabels(T, 'per_site');
    else
        T = addSplitLabels(T, 'global');
    end
    T = addRareEventLabels(T);

    % Explicitly report (not silently drop) any site with zero usable
    % test-split rows -- e.g. a station decommissioned before the global
    % test window begins. This is a real, stated data-coverage limit, not
    % a processing artifact.
    sites = unique(T.Site_ID);
    excludedSites = strings(0);
    for i = 1:numel(sites)
        n = sum(T.Site_ID == sites(i) & T.Split == "test" & ~isnan(T.Power_kW));
        if n == 0
            excludedSites(end+1) = sites(i); %#ok<AGROW>
        end
    end
    if ~isempty(excludedSites)
        fprintf('  NOTE: %d/%d sites have NO valid data in the global test window (likely decommissioned/added outside it): %s\n', ...
            numel(excludedSites), numel(sites), strjoin(excludedSites, ', '));
    end

    outName = strrep(files{f}, '.csv', '_labeled.csv');
    T.Site_ID = "ID_" + T.Site_ID; % prefix guarantees text-type on any future re-read,
                                     % since Ausgrid/PVDAQ site names are pure numbers
                                     % (e.g. "34", "1283") and would otherwise be
                                     % misdetected as a numeric column when reloaded
    writetable(T, fullfile(outDir, outName));

    % Summary
    fprintf('  Rows: %d | Sites: %d\n', height(T), numel(unique(T.Site_ID)));
    splitCats = unique(T.Split);
    for c = 1:numel(splitCats)
        n = sum(T.Split == splitCats(c));
        fprintf('    %s: %d (%.1f%%)\n', splitCats(c), n, 100*n/height(T));
    end
    fprintf('  Rare events flagged: %d (%.2f%%)\n', sum(T.IsRareEvent), 100*mean(T.IsRareEvent));
end

fprintf('\nPhase 2 done. Labeled files written to: %s\n', outDir);

%% ---- Helper functions ----

function T = addSplitLabels(T, strategy)
    % strategy = 'global': single calendar-date 70/15/15 split shared by
    %   all sites (fair for datasets where every site covers the same
    %   installation timeline -- DKASC, HKUST, Ausgrid).
    % strategy = 'per_site': each site split 70/15/15 on its OWN
    %   timeline independently (needed when sites have very different
    %   installation/decommission dates -- PVDAQ).
    if strcmp(strategy, 'global')
        tMin = min(T.Timestamp);
        tMax = max(T.Timestamp);
        totalSpan = tMax - tMin;
        trainCutoff = tMin + 0.70 * totalSpan;
        valCutoff   = tMin + 0.85 * totalSpan;

        T.Split = repmat("train", height(T), 1);
        T.Split(T.Timestamp > trainCutoff & T.Timestamp <= valCutoff) = "val";
        T.Split(T.Timestamp > valCutoff) = "test";
    else % per_site
        T.Split = repmat("train", height(T), 1);
        sites = unique(T.Site_ID);
        for i = 1:numel(sites)
            idx = find(T.Site_ID == sites(i));
            [~, order] = sort(T.Timestamp(idx));
            idxSorted = idx(order);
            n = numel(idxSorted);
            nTrain = round(0.70 * n);
            nVal   = round(0.15 * n);
            T.Split(idxSorted(1:nTrain)) = "train";
            T.Split(idxSorted(nTrain+1 : nTrain+nVal)) = "val";
            T.Split(idxSorted(nTrain+nVal+1 : end)) = "test";
        end
    end
end

function T = addRareEventLabels(T)
    % Flags rows as rare/extreme events using generic, dataset-agnostic
    % rules (works even when irradiance/weather columns are all-NaN for
    % a given dataset, e.g. Ausgrid):
    %   1. Clipping: Power_kW within 1% of that site's observed max (P99.5)
    %      for at least 3 consecutive intervals -- proxy for inverter clipping.
    %   2. Low-irradiance / cloud-transient: daytime hours (irradiance or
    %      power > 0) where Power_kW drops >50% from the previous reading
    %      within a single step -- proxy for sudden cloud cover.
    %   3. Missing/near-zero-variance stretches are NOT flagged here (handled
    %      separately as a data-quality issue, not a "rare event").
    T.IsRareEvent = false(height(T), 1);
    sites = unique(T.Site_ID);
    for i = 1:numel(sites)
        idx = find(T.Site_ID == sites(i));
        [~, order] = sort(T.Timestamp(idx));
        idxSorted = idx(order);
        p = T.Power_kW(idxSorted);

        % --- Clipping proxy ---
        pmax = prctile(p(p > 0), 99.5);
        if ~isnan(pmax) && pmax > 0
            nearMax = p >= 0.99 * pmax;
            clipRun = movsum(nearMax, [2 0]) >= 3; % 3+ consecutive near-max points
            T.IsRareEvent(idxSorted(clipRun)) = true;
        end

        % --- Sudden-drop (cloud transient) proxy ---
        dp = [0; diff(p)];
        suddenDrop = p > 0 & [false; p(1:end-1) > 0] & (dp ./ max(p, eps)) < -0.5;
        T.IsRareEvent(idxSorted(suddenDrop)) = true;
    end
end
