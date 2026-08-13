function inspect_HKUST_sensors(folderPath)
% INSPECT_HKUST_SENSORS Prints header + first 2 data rows for every file
% in every Meteorological sensor subfolder, so column names/formats can
% be confirmed before writing the real importer.
%
% folderPath: path to "2_HKUST_Rooftop_HK" folder

metPath = fullfile(folderPath, 'Time series dataset', 'Meteorological dataset');
sensorFolders = {'Irradiance','Rainfall','Relative Humidity', ...
                 'Sea Level Pressure','Temperature','Visibility','Wind'};

for s = 1:numel(sensorFolders)
    sPath = fullfile(metPath, sensorFolders{s});
    if ~isfolder(sPath)
        fprintf('=== %s: FOLDER NOT FOUND ===\n\n', sensorFolders{s});
        continue
    end
    files = [dir(fullfile(sPath, '*.csv')); dir(fullfile(sPath, '*.xlsx'))];
    fprintf('=== %s (%d files) ===\n', sensorFolders{s}, numel(files));
    for i = 1:numel(files)
        f = fullfile(files(i).folder, files(i).name);
        try
            [~,~,ext] = fileparts(f);
            if strcmpi(ext, '.xlsx')
                Ti = readtable(f, 'VariableNamingRule', 'preserve');
            else
                Ti = readtable(f, 'VariableNamingRule', 'preserve', 'Delimiter', ',');
            end
            fprintf('  %s | columns: %s\n', files(i).name, strjoin(Ti.Properties.VariableNames, ' | '));
            disp(Ti(1:min(2,height(Ti)), :));
        catch ME
            fprintf('  %s | READ ERROR: %s\n', files(i).name, ME.message);
        end
    end
    fprintf('\n');
end

end
