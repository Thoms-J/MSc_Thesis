% Consec_Batch_PlantMove.m
% Script to consecutively compare .las files using PlantMove_matlab

folderPath = 'path_to_las_files';
lasFiles = dir(fullfile(folderPath, '*.las'));
if numel(lasFiles) < 2
    error('At least two .las files are required.');
end

% Sort files by the number in the filename (chunk number at start)
fileNames = {lasFiles.name};
chunkNumbers = zeros(size(fileNames));

for i = 1:length(fileNames)
    tokens = regexp(fileNames{i}, '(\d+)', 'tokens');
    if ~isempty(tokens)
        % Use the FIRST number group (leading index like 00, 01, 02, ...)
        chunkNumbers(i) = str2double(tokens{1}{1});
    else
        error('No numeric identifier found in filename: %s', fileNames{i});
    end
end

[~, sortedIdx] = sort(chunkNumbers);
lasFiles = lasFiles(sortedIdx);

% Parameters for PlantMove_matlab
denoise   = 4;
dsp       = 3;
k         = 5;  %10 for large trees
plotflag  = 0;  % Disable plotting

% Loop through consecutive pairs
for i = 1:numel(lasFiles) - 1
    % Build paths
    lasPathX = fullfile(folderPath, lasFiles(i).name);
    lasPathY = fullfile(folderPath, lasFiles(i+1).name);

    % Debug check
    fprintf('Loading file: %s\n', lasPathX);
    if ~isfile(lasPathX)
        error('File not found: %s', lasPathX);
    end

    % Load file i (X)
    Xlas = lasdata(lasPathX);
    X = [Xlas.x(:), Xlas.y(:), Xlas.z(:)];
    X = double(unique(X,'rows')); % ensure double

    % Load file i+1 (Y)
    fprintf('Loading file: %s\n', lasPathY);
    if ~isfile(lasPathY)
        error('File not found: %s', lasPathY);
    end
    Ylas = lasdata(lasPathY);
    Y = [Ylas.x(:), Ylas.y(:), Ylas.z(:)];
    Y = double(unique(Y,'rows')); % ensure double
    
    % Call original function
    motionfield = PlantMove_matlab(X, Y, denoise, dsp, k, plotflag);

    % Save output
    [~, xName, ~] = fileparts(lasFiles(i).name);
    [~, yName, ~] = fileparts(lasFiles(i+1).name);
    outputFile = fullfile(folderPath, sprintf('Consec_%s_to_%s.txt', xName, yName));
    writematrix(motionfield, outputFile, 'Delimiter', ' ');

    fprintf('Processed: %s -> %s -> Output: %s\n', lasFiles(i).name, lasFiles(i+1).name, outputFile);
end

fprintf('All consecutive file pairs processed.\n');
