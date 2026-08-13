function T = import_4_PVDAQ(folderPath, systemIDs)
% IMPORT_4_PVDAQ Reads NREL PVDAQ dataset (Dataset 4), integrated version.
%
% folderPath: path to "4_PVDAQ_NREL_US" folder containing:
%   system_id=<ID>/year=<YYYY>/month=<M>/day=<D>/system_<ID>__date_<YYYY_MM_DD>.csv
%
% systemIDs: cell array of system_id strings, e.g. {'4','1367',...}.
%            If omitted, imports ALL system_id folders found (very large).
%
% This version fixes, in one integrated pass, every issue found across
% individual-system testing:
%   1. UNITS: raw power columns are NOT consistently in kW or W across
%      systems (confirmed: system 4 was in kW-scale, system 1367 was in
%      W-scale for the same-named column). Auto-detected per system using
%      the official NREL dc_capacity_kW from systems_20250729.csv as a
%      reference scale, rather than assuming a fixed unit.
%   2. MULTIPLE POWER SENSORS: some systems have more than one ac_power/
%      dc_power channel (e.g. multiple inverters) -- ALL matching columns
%      are summed for total site power, not just the first match.
%   3. IRRADIANCE DROPPED BY DATASTORE: tabularTextDatastore silently
%      drops columns inconsistent across a system's daily files. Real
%      irradiance values are recovered via a lightweight per-file
%      header+2-column read (fast: no full-row parsing).
%   4. DUPLICATE TIMESTAMPS: de-duplicated (averaged) before merging, to
%      avoid row multiplication during the irradiance join.
%   5. MISSING POWER COLUMN: systems with no power column are skipped
%      with a warning instead of erroring.
%   6. MIXED COLUMN TYPES (cell vs numeric vs datetime) across files:
%      normalized before any concatenation.
%
% Output: T, a table with common schema columns:
%   Timestamp, Power_kW, Irradiance_Wm2, Site_ID, Climate_Zone

% Reference capacities (dc_capacity_kW) from the official NREL PVDAQ
% systems_20250729.csv metadata, used to auto-detect W vs kW scaling.
capacityLookup = containers.Map();
capacityLookup('4')     = 1.0;
capacityLookup('1283')  = 408.24;
capacityLookup('34')    = 146.64;
capacityLookup('1367')  = 277.16;
capacityLookup('4901')  = 242.5;
capacityLookup('1199')  = 52.92;
capacityLookup('1239')  = 20.16;
capacityLookup('1422')  = 6.0;
capacityLookup('2105')  = 110.0;
capacityLookup('10137') = 3.44;

if nargin < 2 || isempty(systemIDs)
    d = dir(fullfile(folderPath, 'system_id=*'));
    systemIDs = extractAfter({d.name}, 'system_id=');
end

allParts = {};
for s = 1:numel(systemIDs)
    sysID = char(systemIDs{s});
    sysFolder = fullfile(folderPath, ['system_id=' sysID]);
    csvFiles = dir(fullfile(sysFolder, '**', '*.csv'));
    fprintf('System %s: %d daily files found...\n', sysID, numel(csvFiles));

    if isempty(csvFiles)
        warning('System %s: no files found, skipping.', sysID);
        continue
    end

    fileList = fullfile({csvFiles.folder}, {csvFiles.name});

    % --- Fast bulk read (power + timestamp; irradiance recovered separately) ---
    ds = tabularTextDatastore(fileList, 'VariableNamingRule', 'preserve');
    Tsys = readall(ds);
    if isempty(Tsys)
        warning('System %s: datastore read returned empty, skipping.', sysID);
        continue
    end

    ts_col = detectCol(Tsys, {'measured_on','timestamp','Timestamp','Date_Time','DateTime'});
    if isempty(ts_col)
        warning('System %s: no timestamp column found, skipping.', sysID);
        continue
    end
    Tsys.Timestamp = datetime(Tsys.(ts_col));

    % Sum ALL power-like columns (multiple inverters/strings), preferring
    % AC power (grid-relevant output) over DC if both present.
    acCols = Tsys.Properties.VariableNames(contains(Tsys.Properties.VariableNames, 'ac_power', 'IgnoreCase', true));
    dcCols = Tsys.Properties.VariableNames(contains(Tsys.Properties.VariableNames, 'dc_power', 'IgnoreCase', true));
    powerCols = acCols;
    if isempty(powerCols)
        powerCols = dcCols;
    end
    if isempty(powerCols)
        warning('System %s: no power column found, skipping.', sysID);
        continue
    end
    rawPower = sum(Tsys{:, powerCols}, 2, 'omitnan');

    % --- Unit auto-detection using official capacity metadata ---
    if isKey(capacityLookup, sysID)
        capKW = capacityLookup(sysID);
        p99 = prctile(rawPower(~isnan(rawPower) & rawPower > 0), 99);
        if ~isnan(p99) && p99 > 3 * capKW
            % Values far exceed plausible kW output for this system's
            % rated capacity -> raw column is actually in Watts.
            Power_kW = rawPower / 1000;
        else
            Power_kW = rawPower;
        end
    else
        % No reference capacity available for this system -- leave as-is
        % but flag it so the discrepancy isn't silently hidden.
        warning('System %s: no capacity reference for unit check -- Power_kW may be mis-scaled.', sysID);
        Power_kW = rawPower;
    end

    % --- De-duplicate timestamps (average if duplicated) before any join ---
    Tpower = table(Tsys.Timestamp, Power_kW, 'VariableNames', {'Timestamp','Power_kW'});
    [uniqueTimes, ~, ic] = unique(Tpower.Timestamp);
    if numel(uniqueTimes) < height(Tpower)
        Power_kW_dedup = accumarray(ic, Tpower.Power_kW, [], @(x) mean(x, 'omitnan'));
        Tpower = table(uniqueTimes, Power_kW_dedup, 'VariableNames', {'Timestamp','Power_kW'});
    end

    % --- Targeted irradiance recovery (lightweight, 2 columns per file only) ---
    irrRows = table();
    for i = 1:numel(fileList)
        try
            optsHeader = detectImportOptions(fileList{i}, 'VariableNamingRule', 'preserve');
        catch
            continue
        end
        irrColName = firstMatch(optsHeader.VariableNames, {'poa_irradiance','ghi','irradiance'});
        tsColName  = firstMatch(optsHeader.VariableNames, {'measured_on','timestamp'});
        if isempty(irrColName) || isempty(tsColName)
            continue
        end
        optsHeader.SelectedVariableNames = {tsColName, irrColName};
        try
            Ti_small = readtable(fileList{i}, optsHeader);
        catch
            continue
        end
        Ti_small.Properties.VariableNames = {'measured_on','Irradiance_Wm2_raw'};
        if iscell(Ti_small.Irradiance_Wm2_raw) || isstring(Ti_small.Irradiance_Wm2_raw)
            Ti_small.Irradiance_Wm2_raw = str2double(Ti_small.Irradiance_Wm2_raw);
        end
        if iscell(Ti_small.measured_on) || isstring(Ti_small.measured_on)
            Ti_small.measured_on = datetime(Ti_small.measured_on);
        end
        irrRows = [irrRows; Ti_small]; %#ok<AGROW>
    end

    if ~isempty(irrRows)
        irrRows.Timestamp = datetime(irrRows.measured_on);
        [uT, ~, uic] = unique(irrRows.Timestamp);
        irrDedup = accumarray(uic, irrRows.Irradiance_Wm2_raw, [], @(x) mean(x, 'omitnan'));
        irrRows = table(uT, irrDedup, 'VariableNames', {'Timestamp','Irradiance_Wm2'});
        Tmerged = outerjoin(Tpower, irrRows, 'Keys', 'Timestamp', 'MergeKeys', true, 'Type', 'left');
    else
        Tmerged = Tpower;
        Tmerged.Irradiance_Wm2 = NaN(height(Tmerged), 1);
    end

    % --- Sanity-clip physically implausible values (sensor errors) ---
    % Irradiance: no real ground sensor exceeds ~1500 W/m2, and cannot be negative.
    badIrr = Tmerged.Irradiance_Wm2 < 0 | Tmerged.Irradiance_Wm2 > 1500;
    Tmerged.Irradiance_Wm2(badIrr) = NaN;

    % Power: allow small negative (nighttime inverter self-consumption) but
    % reject anything beyond 3x the system's rated capacity as a sensor fault.
    if isKey(capacityLookup, sysID)
        capKW = capacityLookup(sysID);
        badPower = Tmerged.Power_kW < -5 | Tmerged.Power_kW > 3 * capKW;
        Tmerged.Power_kW(badPower) = NaN;
    end

    % Drop rows with invalid (NaT) timestamps entirely -- these carry no
    % usable information and would otherwise corrupt resampling later.
    Tmerged = Tmerged(~isnat(Tmerged.Timestamp), :);

    Tmerged.Site_ID = repmat(string(sysID), height(Tmerged), 1);
    allParts{end+1} = Tmerged; %#ok<AGROW>
end

if isempty(allParts)
    T = table();
    warning('No systems produced data.');
    return
end

T = vertcat(allParts{:});
T = sortrows(T, {'Site_ID','Timestamp'});
T.Climate_Zone = repmat("Mixed_US", height(T), 1); % see systems_20250729.csv kg_climate for per-system Koppen zone

end

function col = detectCol(T, candidates)
    col = '';
    for i = 1:numel(candidates)
        idx = strcmpi(T.Properties.VariableNames, candidates{i});
        if any(idx)
            col = T.Properties.VariableNames{idx};
            return
        end
    end
end

function col = firstMatch(varNames, substrings)
    col = '';
    for i = 1:numel(substrings)
        idx = contains(varNames, substrings{i}, 'IgnoreCase', true);
        if any(idx)
            matches = varNames(idx);
            col = matches{1};
            return
        end
    end
end
