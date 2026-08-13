function T = import_1_DKASC(folderPath)
% IMPORT_1_DKASC Reads and merges DKASC Alice Springs dataset (Dataset 1)
%
% folderPath: path to "1_DKASC_AliceSprings_AU" folder containing:
%   87-Site_DKA-M9_A+C-Phases.csv
%   91-Site_DKA-M9_B-Phase.csv
%   101-Site_DKA-WeatherStation.csv
%
% Output: T, a timetable with common schema columns:
%   Timestamp, Power_kW, Irradiance_Wm2, Temp_C, Humidity_pct, Wind_ms,
%   Site_ID, Climate_Zone

siteAC_path  = fullfile(folderPath, '87-Site_DKA-M9_A+C-Phases.csv');
siteB_path   = fullfile(folderPath, '91-Site_DKA-M9_B-Phase.csv');
weather_path = fullfile(folderPath, '101-Site_DKA-WeatherStation.csv');

opts = detectImportOptions(siteAC_path, 'VariableNamingRule', 'preserve');
Tac = readtable(siteAC_path, opts);
Tb  = readtable(siteB_path,  opts);

optsW = detectImportOptions(weather_path, 'VariableNamingRule', 'preserve');
Tw  = readtable(weather_path, optsW);

Tac.Timestamp = datetime(Tac.timestamp, 'InputFormat', 'M/d/yyyy H:mm');
Tb.Timestamp  = datetime(Tb.timestamp,  'InputFormat', 'M/d/yyyy H:mm');
Tw.Timestamp  = datetime(Tw.timestamp,  'InputFormat', 'M/d/yyyy H:mm');

Tac = table2timetable(Tac(:, {'Timestamp','Active_Power'}), 'RowTimes', 'Timestamp');
Tb  = table2timetable(Tb(:,  {'Timestamp','Active_Power'}), 'RowTimes', 'Timestamp');
Tw  = table2timetable(Tw(:,  {'Timestamp','Wind_Speed','Weather_Temperature_Celsius', ...
                               'Weather_Relative_Humidity','Global_Horizontal_Radiation'}), ...
                       'RowTimes', 'Timestamp');

% Force Wind_Speed to numeric regardless of how it was read in (fixes
% cases where mixed formatting causes MATLAB to import it as text/cell)
if iscell(Tw.Wind_Speed)
    Tw.Wind_Speed = str2double(Tw.Wind_Speed);
elseif isstring(Tw.Wind_Speed)
    Tw.Wind_Speed = str2double(Tw.Wind_Speed);
end

Tac.Properties.VariableNames{'Active_Power'} = 'Power_AC_kW';
Tb.Properties.VariableNames{'Active_Power'}  = 'Power_B_kW';

Tsite = synchronize(Tac, Tb, 'union');
Tsite.Power_kW = sum([Tsite.Power_AC_kW, Tsite.Power_B_kW], 2, 'omitnan');

T = synchronize(Tsite(:, 'Power_kW'), Tw, 'union');
T = T(~isnan(T.Power_kW), :);

T.Properties.VariableNames{'Global_Horizontal_Radiation'} = 'Irradiance_Wm2';
T.Properties.VariableNames{'Weather_Temperature_Celsius'} = 'Temp_C';
T.Properties.VariableNames{'Weather_Relative_Humidity'}   = 'Humidity_pct';
T.Properties.VariableNames{'Wind_Speed'}                  = 'Wind_ms';

T.Site_ID = repmat("DKASC_1", height(T), 1);
T.Climate_Zone = repmat("BWh_desert", height(T), 1);

end
