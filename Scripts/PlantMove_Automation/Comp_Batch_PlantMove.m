% Comp_Batch_PlantMove.m
% Script to comparatively process all .las files in folder using PlantMove_matlab

folderPath = 'path_to_las_files';
lasFiles = dir(fullfile(folderPath, '*.las'));
if numel(lasFiles) < 2
    error('At least two .las files are required.');
end

% Sort files by the number in the filename (chunk number)
fileNames = {lasFiles.name};
chunkNumbers = zeros(size(fileNames));

for i = 1:length(fileNames)
    tokens = regexp(fileNames{i}, '(\d+)', 'tokens');
    if ~isempty(tokens)
        chunkNumbers(i) = str2double(tokens{end}{1}); % Use the LAST number
    else
        error('No numeric identifier found in filename: %s', fileNames{i});
    end
end

[~, sortedIdx] = sort(chunkNumbers);
lasFiles = lasFiles(sortedIdx);

% Parameters for PlantMove_matlab
denoise     = 4;
dsp         = 3;
k           = 5;    %10 for large trees
plotflag    = 0;    % Turn off plotting for batch processing

% Read the first file as X

Xlas = lasdata(fullfile(folderPath, lasFiles(1).name));
X = [Xlas.x(:), Xlas.y(:), Xlas.z(:)];
X = double(unique(X,'rows'));% ensure double

% Process each remaining file as Y
for i = 2:numel(lasFiles)

    Ylas = lasdata(fullfile(folderPath, lasFiles(i).name));
    Y = [Ylas.x(:), Ylas.y(:), Ylas.z(:)];
    Y = double(unique(Y,'rows'));% ensure double
    
    % Call original function
    motionfield = PlantMove_matlab(X, Y, denoise, dsp, k, plotflag);
    
    % Save motion field to txt file
    [~, xName, ~] = fileparts(lasFiles(1).name);
    [~, yName, ~] = fileparts(lasFiles(i).name);
    outputFile = fullfile(folderPath, sprintf('Compare_%s_to_%s.txt', xName, yName));
    writematrix(motionfield, outputFile, 'Delimiter', ' ');
    
    fprintf('Processed: %s -> Output: %s\n', lasFiles(i).name, outputFile);
end

fprintf('All files processed.\n');

