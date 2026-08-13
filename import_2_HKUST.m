function T = import_2_HKUST(folderPath)
% IMPORT_2_HKUST Reads and merges HKUST Rooftop dataset (Dataset 2)
%
% Confirmed schema (verified against real file headers):
%   Irradiance:         Time, Irradiance (W/m2)
%   Rainfall:           Time, Rainfall(mm)              [2021 file is .xlsx, others .csv]
%   Relative Humidity:  Time, RH (%)
%   Sea Level Pressure: Time, SLP (hPa)
%   Temperature:        Time, Temp (Degree Celsius)
%   Visibility:         Time, Vis (km)
%   Wind:               Time, Wind Speed (m/s), Wind Direction (degree)
%
% Time format in .csv files: 'yyyy/M/d H:mm' (read as text/cell)
% Time format in .xlsx files: already datetime
%
% Output: T, a timetable with common schema columns:
%   Timestamp, Power_kW, Irradiance_Wm2, Temp_C, Humidity_pct, Wind_ms,
%   Site_ID, Climate_Zone
% (Rainfall, SLP, Visibility, Wind Direction are also kept as extra columns)

tsPath  = fullfile(folderPath, 'Time series dataset');
metPath = fullfile(tsPath, 'Meteorological dataset');
pvPath  = fullfile(tsPath, 'PV generation dataset');

sensorFolders = {'Irradiance','Rainfall','Relative Humidity', ...
                 'Sea Level Pressure','Temperature','Visibility','Wind'};

Tw = timetable.empty;
for s = 1:numel(sensorFolders)
    sPath = fullfile(metPath, sensorFolders{s});
    if ~isfolder(sPath)
        warning('Sensor folder not found: %s -- skipping', sPath);
        continue
    end
    files = [dir(fullfile(sPath, '*.csv')); dir(fullfile(sPath, '*.xlsx'))];
    Tsensor = table();
    for i = 1:numel(files)
        f = fullfile(files(i).folder, files(i).name);
        [~,~,ext] = fileparts(f);
        if strcmpi(ext, '.xlsx')
            Ti = readtable(f, 'VariableNamingRule', 'preserve');
        else
            Ti = readtable(f, 'VariableNamingRule', 'preserve', 'Delimiter', ',');
        end
        % Normalize Time to datetime PER FILE before stacking, since some
        % files (.xlsx) read Time as datetime already and others (.csv)
        % read it as cell-string -- vertcat fails on mismatched types.
        if iscell(Ti.Time)
            Ti.Time = datetime(Ti.Time, 'InputFormat', 'yyyy/M/d H:mm');
        elseif isstring(Ti.Time)
            Ti.Time = datetime(Ti.Time, 'InputFormat', 'yyyy/M/d H:mm');
        end
        % (if already datetime, leave as-is)
        Tsensor = [Tsensor; Ti]; %#ok<AGROW>
    end
    if isempty(Tsensor)
        continue
    end

    Tsensor.Timestamp = Tsensor.Time;
    valueCols = setdiff(Tsensor.Properties.VariableNames, {'Time','Timestamp'}, 'stable');
    Tsensor = table2timetable(Tsensor(:, ['Timestamp', valueCols]), 'RowTimes', 'Timestamp');

    if isempty(Tw)
        Tw = Tsensor;
    else
        Tw = synchronize(Tw, Tsensor, 'union');
    end
end

% ---- PV generation: site-level, both optimizer branches ----
siteWith    = fullfile(pvPath, 'PV stations with panel level optimizer', 'Site level dataset');
siteWithout = fullfile(pvPath, 'PV stations without panel level optimizer', 'Site level dataset');

siteFiles = [dir(fullfile(siteWith, '*.csv')); dir(fullfile(siteWithout, '*.csv'))];

Tpv = table();
for i = 1:numel(siteFiles)
    f = fullfile(siteFiles(i).folder, siteFiles(i).name);
    Ti = readtable(f, 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    ts_col = detectTimestampCol(Ti);
    if iscell(Ti.(ts_col))
        Ti.Timestamp = datetime(Ti.(ts_col), 'InputFormat', 'yyyy/M/d H:mm');
    else
        Ti.Timestamp = datetime(Ti.(ts_col));
    end
    powerCol = Ti.Properties.VariableNames(contains(Ti.Properties.VariableNames, {'Power','power','kW'}));
    if isempty(powerCol)
        continue
    end
    [~, stationName] = fileparts(siteFiles(i).name);
    Ti.Power_kW = Ti.(powerCol{1});
    Ti.Site_ID = repmat(string(stationName), height(Ti), 1);
    Tpv = [Tpv; Ti(:, {'Timestamp','Power_kW','Site_ID'})]; %#ok<AGROW>
end
Tpv = sortrows(Tpv, 'Timestamp');

% ---- Merge PV + weather ----
% Tpv has MULTIPLE rows per timestamp (one per station), so we cannot use
% synchronize/timetable union (which requires unique times per input).
% Instead, do a table JOIN on Timestamp: this correctly broadcasts the
% single shared weather reading across every station row at that time.
Tw_table = timetable2table(Tw);
T = outerjoin(Tpv, Tw_table, 'Keys', 'Timestamp', 'MergeKeys', true, 'Type', 'left');
T = sortrows(T, 'Timestamp');

% Rename to common schema (exact original names confirmed from real files)
T = renameIfExists(T, 'Irradiance (W/m2)', 'Irradiance_Wm2');
T = renameIfExists(T, 'Temp (Degree Celsius)', 'Temp_C');
T = renameIfExists(T, 'RH (%)', 'Humidity_pct');
T = renameIfExists(T, 'Wind Speed (m/s)', 'Wind_ms');
T = renameIfExists(T, 'Wind Direction (degree)', 'Wind_Direction_deg');
T = renameIfExists(T, 'Rainfall(mm)', 'Rainfall_mm');
T = renameIfExists(T, 'SLP (hPa)', 'Pressure_hPa');
T = renameIfExists(T, 'Vis (km)', 'Visibility_km');

T.Climate_Zone = repmat("Cwa_subtropical", height(T), 1);

end

function col = detectTimestampCol(T)
    candidates = {'timestamp','Timestamp','Time','date_time','Date_Time','DateTime'};
    col = '';
    for i = 1:numel(candidates)
        if any(strcmpi(T.Properties.VariableNames, candidates{i}))
            col = T.Properties.VariableNames{strcmpi(T.Properties.VariableNames, candidates{i})};
            return
        end
    end
    error('Timestamp column not found -- columns present: %s', strjoin(T.Properties.VariableNames, ', '));
end

function T = renameIfExists(T, oldName, newName)
    if any(strcmp(T.Properties.VariableNames, oldName))
        T.Properties.VariableNames{oldName} = newName;
    end
end
