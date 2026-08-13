%% run_harmonization.m
% Phase 1 driver: imports all 4 SolarBench datasets, harmonizes them to a
% common schema, resamples to 15-min resolution, and writes one
% harmonized CSV per dataset.
%
% Common schema columns:
%   Timestamp, Power_kW, Irradiance_Wm2, Temp_C, Humidity_pct, Wind_ms,
%   Site_ID, Climate_Zone
% (Datasets lacking a given sensor get that column filled with NaN --
% e.g. Ausgrid has no on-site weather; this PVDAQ subset has no
% temp/humidity/wind, only power + irradiance.)

clear; clc;

baseDir = 'C:\Users\Shaho\Desktop\claude_projects\R9\pv_forecasting_enchmark\dataset';
outDir  = fullfile(baseDir, 'harmonized');
if ~exist(outDir, 'dir'); mkdir(outDir); end

commonCols = {'Timestamp','Power_kW','Irradiance_Wm2','Temp_C','Humidity_pct','Wind_ms','Site_ID','Climate_Zone'};

%% Dataset 1: DKASC
fprintf('--- Dataset 1: DKASC ---\n');
T1 = import_1_DKASC(fullfile(baseDir, '1_DKASC_AliceSprings_AU'));
T1 = timetable2table(T1); % T1 comes back as a timetable; make Timestamp an explicit column
T1 = padToCommonSchema(T1, commonCols);
writetable(T1, fullfile(outDir, 'dataset1_DKASC_raw_harmonized.csv'));
T1t = table2timetable(T1, 'RowTimes', 'Timestamp');
numericCols1 = varfun(@isnumeric, T1t, 'OutputFormat', 'uniform');
numericColNames1 = T1t.Properties.VariableNames(numericCols1);
T1r = retime(T1t(:, numericColNames1), 'regular', 'mean', 'TimeStep', minutes(15));
T1r.Site_ID = repmat(T1.Site_ID(1), height(T1r), 1);
T1r.Climate_Zone = repmat(T1.Climate_Zone(1), height(T1r), 1);
writetimetable(T1r, fullfile(outDir, 'dataset1_DKASC_15min.csv'));
fprintf('  -> %d rows written (15-min)\n', height(T1r));

%% Dataset 2: HKUST
fprintf('--- Dataset 2: HKUST ---\n');
T2 = import_2_HKUST(fullfile(baseDir, '2_HKUST_Rooftop_HK'));
T2 = padToCommonSchema(T2, commonCols);
writetable(T2, fullfile(outDir, 'dataset2_HKUST_raw_harmonized.csv'));
% Multiple sites share timestamps -- retime per site to avoid mixing stations
T2r = retimePerSite(T2, minutes(15));
writetable(T2r, fullfile(outDir, 'dataset2_HKUST_15min.csv'));
fprintf('  -> %d rows written (15-min, per-site)\n', height(T2r));

%% Dataset 3: Ausgrid
fprintf('--- Dataset 3: Ausgrid ---\n');
T3 = import_3_Ausgrid(fullfile(baseDir, '3_Ausgrid_Sydney_AU'));
T3 = padToCommonSchema(T3, commonCols);
writetable(T3, fullfile(outDir, 'dataset3_Ausgrid_raw_harmonized.csv'));
% Ausgrid is natively half-hourly -- no upsampling to 15-min (would fabricate data)
fprintf('  -> %d rows written (native half-hourly, no resample)\n', height(T3));

%% Dataset 4: PVDAQ (8 verified systems)
fprintf('--- Dataset 4: PVDAQ ---\n');
T4 = import_4_PVDAQ(fullfile(baseDir, '4_PVDAQ_NREL_US'), ...
    {'4','1283','34','1367','4901','1199','1239','1422'});
T4 = padToCommonSchema(T4, commonCols);
writetable(T4, fullfile(outDir, 'dataset4_PVDAQ_raw_harmonized.csv'));
T4r = retimePerSite(T4, minutes(15));
writetable(T4r, fullfile(outDir, 'dataset4_PVDAQ_15min.csv'));
fprintf('  -> %d rows written (15-min, per-site)\n', height(T4r));

fprintf('\nDone. Harmonized files written to: %s\n', outDir);

%% ---- Helper functions ----

function T = padToCommonSchema(T, commonCols)
    % Ensure T has every common-schema column, filling missing ones with NaN
    % (or <missing> for string columns), in a fixed column order.
    for i = 1:numel(commonCols)
        c = commonCols{i};
        if ~ismember(c, T.Properties.VariableNames)
            if strcmp(c, 'Site_ID') || strcmp(c, 'Climate_Zone')
                T.(c) = repmat("", height(T), 1);
            else
                T.(c) = NaN(height(T), 1);
            end
        end
    end
    T = T(:, commonCols);
end

function Tout = retimePerSite(T, step)
    % Retime a long-format multi-site table (Timestamp, ..., Site_ID) to a
    % fixed time step WITHIN each site separately, then recombine, so
    % different stations' timestamps never get averaged together.
    sites = unique(T.Site_ID);
    parts = cell(numel(sites), 1);
    for i = 1:numel(sites)
        Tsite = T(T.Site_ID == sites(i), :);
        Tsite = sortrows(Tsite, 'Timestamp');
        [~, ia] = unique(Tsite.Timestamp);
        Tsite = Tsite(ia, :); % drop any remaining duplicate timestamps
        numericCols = varfun(@isnumeric, Tsite, 'OutputFormat', 'uniform');
        numericColNames = Tsite.Properties.VariableNames(numericCols);
        Tt = table2timetable(Tsite(:, ['Timestamp', numericColNames]), 'RowTimes', 'Timestamp');
        Tr = retime(Tt, 'regular', 'mean', 'TimeStep', step);
        Tr = timetable2table(Tr);
        Tr.Site_ID = repmat(sites(i), height(Tr), 1);
        parts{i} = Tr;
    end
    Tout = vertcat(parts{:});
end
