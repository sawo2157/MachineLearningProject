%% =======================
% 1. LOAD DATA
% ========================

data = readtable('C:\Users\sam01\Documents\MATLAB\MCEN3030\MachineLearningProject\dataset_02052023.csv');

%% =======================
% 2. REMOVE INVALID SENSOR ROWS
% ========================

% Define sensor columns (core features that must exist)
sensorVars = { ...
    'Current_J0','Temperature_T0',...
    'Current_J1','Temperature_J1',...
    'Current_J2','Temperature_J2',...
    'Current_J3','Temperature_J3',...
    'Current_J4','Temperature_J4',...
    'Current_J5','Temperature_J5',...
    'Speed_J0','Speed_J1','Speed_J2','Speed_J3','Speed_J4','Speed_J5',...
    'Tool_current'};

% Remove rows where ALL sensor values are missing
data = data(~all(ismissing(data(:, sensorVars)),2), :);

%% =======================
% 3. CREATE TARGET VARIABLE
% ========================

% Convert to logical
ps = strcmpi(string(data.Robot_ProtectiveStop), 'true');
gl = strcmpi(string(data.grip_lost), 'true');

data.Failure = ps | gl;

%% =======================
% 4. SORT BY TIME
% ========================

data = sortrows(data, 'Num');

%% =======================
% 5. FEATURE ENGINEERING (TRENDS)
% ========================

trendVars = { ...
    'Current_J0','Current_J1','Current_J2','Current_J3','Current_J4','Current_J5',...
    'Temperature_T0','Temperature_J1','Temperature_J2','Temperature_J3','Temperature_J4','Temperature_J5',...
    'Tool_current'};

windowSize = 5;

for i = 1:length(trendVars)
    f = trendVars{i};
    
    % Delta (1-second difference)
    data.([f '_delta']) = [NaN; diff(data.(f))];
    
    % Rolling mean (5-second window)
    data.([f '_rollmean']) = movmean(data.(f), windowSize, 'Endpoints','shrink');
end

%% =======================
% 6. REMOVE NaNs FROM FEATURE ENGINEERING
% ========================

data = rmmissing(data);

%% =======================
% 7. PREPARE FEATURES
% ========================

predictorVars = data.Properties.VariableNames;

predictorVars = setdiff(predictorVars, { ...
    'Robot_ProtectiveStop','grip_lost','Failure',...
    'Timestamp','cycle','Num'}); % remove non-feature columns

X = data{:, predictorVars};
Y = data.Failure;

%% =======================
% 8. TIME-BASED 70/30 SPLIT
% ========================

n = height(data);
splitIdx = floor(0.7 * n);

X_train = X(1:splitIdx, :);
Y_train = Y(1:splitIdx);

X_test  = X(splitIdx+1:end, :);
Y_test  = Y(splitIdx+1:end);

%% =======================
% 9. TRAIN RANDOM FOREST
% ========================

numTrees = 100;

model = TreeBagger(numTrees, X_train, Y_train, ...
    'Method','classification', ...
    'OOBPrediction','On', ...
    'OOBPredictorImportance','on');

%% =======================
% 10. PREDICTIONS
% ========================

Y_pred = predict(model, X_test);
Y_pred = str2double(Y_pred);
Y_pred = logical(Y_pred);

%% =======================
% 11. CONFUSION MATRIX
% ========================

figure;

Y_test_lbl = categorical(Y_test, [0 1], {'No Failure (0)', 'Failure (1)'});
Y_pred_lbl = categorical(Y_pred, [0 1], {'No Failure (0)', 'Failure (1)'});

cm = confusionchart(Y_test_lbl, Y_pred_lbl);

cm.Title = 'Confusion Matrix';
cm.XLabel = 'Predicted Class';
cm.YLabel = 'Actual Class';

%% ===================
%Accuracy Summary
fprintf('Acc: %.3f | Test failure rate: %.3f | Pred failure rate: %.3f | OOB error: %.3f\n', mean(Y_pred==Y_test), mean(Y_test), mean(Y_pred), oobError(model,'Mode','ensemble'))


%% =======================
% 12. FEATURE IMPORTANCE
% ========================

importance = model.OOBPermutedPredictorDeltaError;

% Sort descending
[sortedImp, idx] = sort(importance, 'descend');
sortedNames = predictorVars(idx);

% -----------------------
% Clean feature names
% -----------------------
cleanNames = strings(size(sortedNames));

for i = 1:length(sortedNames)
    name = sortedNames{i};

    jointID = "";

    % Detect joint number
    tokens = regexp(name, '(J|T)(\d)', 'tokens');
    if ~isempty(tokens)
        jointID = tokens{1}{2};
    end

    % Base naming
    if contains(name, 'Temperature')
        base = "Temp J" + jointID;
    elseif contains(name, 'Current')
        base = "Current J" + jointID;
    elseif contains(name, 'Speed')
        base = "Speed J" + jointID;
    elseif contains(name, 'Tool_current')
        base = "Tool Current";
    else
        base = name;
    end

    % Trend suffix
    if contains(name, 'delta')
        base = base + " Δ";
    elseif contains(name, 'rollmean')
        base = base + " R.M.";
    end

    cleanNames(i) = base;
end

% -----------------------
% Assign colors per joint
% -----------------------

colors = lines(7); % J0–J5 + placeholder

toolColor = [0 0 0]; % FIXED: true black

barColors = zeros(length(sortedNames),3);

for i = 1:length(sortedNames)
    name = sortedNames{i};

    if contains(name, 'Tool_current')
        barColors(i,:) = toolColor;
    else
        tokens = regexp(name, '(J|T)(\d)', 'tokens');
        if ~isempty(tokens)
            j = str2double(tokens{1}{2}) + 1;
            barColors(i,:) = colors(j,:);
        else
            barColors(i,:) = [0.6 0.6 0.6];
        end
    end
end

% -----------------------
% Plot
% -----------------------
figure;
b = barh(sortedImp);

b.FaceColor = 'flat';
b.CData = barColors;

set(gca, 'YDir','reverse');
yticks(1:length(cleanNames));
yticklabels(cleanNames);
title('Feature Importance (OOB Permutation)');
xlabel('Importance Score');
set(gca, 'FontSize', 11);

% -----------------------
% SINGLE UNIFIED LEGEND
% -----------------------

hold on;

h = gobjects(9,1);

jointColors = colors(1:6,:);
jointColors = [jointColors; toolColor];

jointLabels = {
    'J0','J1','J2','J3','J4','J5','Tool Current'
};

for i = 1:7
    h(i) = plot(nan,nan,'s', ...
        'MarkerFaceColor', jointColors(i,:), ...
        'MarkerEdgeColor', jointColors(i,:), ...
        'MarkerSize', 8);
end

% Δ and R.M.
h(8) = plot(nan,nan,'k-','LineWidth',1.5);
h(9) = plot(nan,nan,'k--','LineWidth',1.5);

legendLabels = [
    jointLabels, ...
    {'Δ = 1-second change', 'R.M. = 5-second rolling mean'}
];

legend(h, legendLabels, 'Location','best');

TP = sum((Y_pred==1) & (Y_test==1));
FP = sum((Y_pred==1) & (Y_test==0));
FN = sum((Y_pred==0) & (Y_test==1));

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);

fprintf('\nPrecision: %.3f\n', precision);
fprintf('Recall (FAILURE DETECTION): %.3f\n', recall);
fprintf('Predicted failure rate: %.3f\n', mean(Y_pred));