function T = import_3_Ausgrid(folderPath)
% IMPORT_3_AUSGRID Reads and reshapes Ausgrid Sydney dataset (Dataset 3)
%
% folderPath: path to "3_Ausgrid_Sydney_AU" folder containing:
%   Solar home 2010-2011.csv
%   Solar home 2011-2012.csv
%   Solar home 2012-2013.csv
%
% Known Ausgrid wide format per row: Customer, Generator Capacity, Postcode,
% Consumption Category, date, then 48 half-hourly columns.
% This version is VECTORIZED (no per-row loop) for speed on 300 customers x 3 years.
%
% Output: T, a table with common schema columns:
%   Timestamp, Power_kW, Site_ID, Climate_Zone
% NOTE: Ausgrid has NO on-site irradiance/temp/humidity/wind.

files = {'Solar home 2010-2011.csv', 'Solar home 2011-2012.csv', 'Solar home 2012-2013.csv'};

allParts = {};
for i = 1:numel(files)
    f = fullfile(folderPath, files{i});
    if ~isfile(f)
        warning('File not found: %s -- skipping', f);
        continue
    end
    opts = detectImportOptions(f, 'VariableNamingRule', 'preserve', 'NumHeaderLines', 1);
    dateColName = opts.VariableNames{contains(opts.VariableNames, 'date', 'IgnoreCase', true)};
    opts = setvartype(opts, dateColName, 'string'); % force text, avoid auto datetime century bug
    Traw = readtable(f, opts);

    idCol   = Traw.Properties.VariableNames{contains(Traw.Properties.VariableNames, 'Customer', 'IgnoreCase', true)};
    catCol  = Traw.Properties.VariableNames{contains(Traw.Properties.VariableNames, 'Consumption Category', 'IgnoreCase', true)};
    dateCol = dateColName;

    isGen = strcmpi(string(Traw.(catCol)), 'GC');
    Tg = Traw(isGen, :);
    if isempty(Tg)
        continue
    end

    metaExclude = {idCol, catCol, dateCol, 'Postcode', 'Generator Capacity', 'Row Quality'};
    hhCols = Tg.Properties.VariableNames;
    for m = 1:numel(metaExclude)
        hhCols = hhCols(~contains(hhCols, metaExclude{m}, 'IgnoreCase', true));
    end

    nHH = numel(hhCols);
    nRows = height(Tg);

    dates = parseAusgridDate(Tg.(dateCol));
    halfHourMinutes = (0:nHH-1) * 30; % 0, 30, 60, ... minutes after midnight

    % Vectorized reshape: repeat each date across 48 half-hours, and tile
    % the half-hour offsets across all rows, instead of looping row-by-row
    dateRep = repelem(dates, nHH);                 % [nRows*nHH x 1]
    minuteRep = repmat(halfHourMinutes(:), nRows, 1); % [nRows*nHH x 1]
    Timestamp = dateRep + minutes(minuteRep);

    energy_kWh = reshape(Tg{:, hhCols}', [], 1); % row-major flatten to match Timestamp order
    Power_kW = energy_kWh / 0.5;

    Site_ID = repelem(string(Tg.(idCol)), nHH);

    partT = table(Timestamp, Power_kW, Site_ID);
    allParts{end+1} = partT; %#ok<AGROW>
end

T = vertcat(allParts{:});
T = sortrows(T, {'Site_ID','Timestamp'});
T.Climate_Zone = repmat("Cfa_temperate", height(T), 1);

end

function dt = parseAusgridDate(dateStrings)
    % Ausgrid's 3 yearly files use different date formats:
    %   2010-2011.csv -> 'd-MMM-yy'   (e.g. '1-Jul-10')
    %   2011-2012.csv -> 'd/MM/yyyy'  (e.g. '1/07/2011')
    %   2012-2013.csv -> 'd/MM/yyyy'  (e.g. '1/07/2012')
    % Try each known format until one parses the whole column successfully.
    formats = {'d-MMM-yy', 'd/MM/yyyy', 'dd/MM/yyyy'};
    dt = NaT(size(dateStrings));
    lastErr = [];
    for i = 1:numel(formats)
        try
            dt = datetime(dateStrings, 'InputFormat', formats{i});
            return
        catch ME
            lastErr = ME;
        end
    end
    rethrow(lastErr);
end
