%% ======================= 
% 1. LOAD DATA
% ========================

data = readtable('C:\Users\sam01\Documents\MATLAB\MCEN3030\MachineLearningProject\dataset_02052023.csv');

%% =======================
% 2. REMOVE INVALID SENSOR ROWS
% ========================

sensorVars = { ...
    'Current_J0','Temperature_T0',...
    'Current_J1','Temperature_J1',...
    'Current_J2','Temperature_J2',...
    'Current_J3','Temperature_J3',...
    'Current_J4','Temperature_J4',...
    'Current_J5','Temperature_J5',...
    'Speed_J0','Speed_J1','Speed_J2','Speed_J3','Speed_J4','Speed_J5',...
    'Tool_current'};

data = data(~all(ismissing(data(:, sensorVars)),2), :);

%% =======================
% 3. CREATE TARGET VARIABLE
% ========================

ps = strcmpi(string(data.Robot_ProtectiveStop), 'true');
gl = strcmpi(string(data.grip_lost), 'true');

data.Failure = ps | gl;

%% =======================
% 4. SORT BY TIME
% ========================

data = sortrows(data, 'Num');

%% =======================
% 5. FEATURE ENGINEERING
% ========================

trendVars = { ...
    'Current_J0','Current_J1','Current_J2','Current_J3','Current_J4','Current_J5',...
    'Temperature_T0','Temperature_J1','Temperature_J2','Temperature_J3','Temperature_J4','Temperature_J5',...
    'Tool_current'};

windowSize = 5;

for i = 1:length(trendVars)
    f = trendVars{i};
    data.([f '_delta']) = [NaN; diff(data.(f))];
    data.([f '_rollmean']) = movmean(data.(f), windowSize, 'Endpoints','shrink');
end

data.TotalCurrent = sum([data.Current_J0,data.Current_J1,data.Current_J2,...
                         data.Current_J3,data.Current_J4,data.Current_J5],2);

data.CurrentImbalance = var([data.Current_J0,data.Current_J1,data.Current_J2,...
                             data.Current_J3,data.Current_J4,data.Current_J5],0,2);

data.TotalTemp = sum([data.Temperature_T0,data.Temperature_J1,data.Temperature_J2,...
                      data.Temperature_J3,data.Temperature_J4,data.Temperature_J5],2);

data.MechanicalStress = data.TotalCurrent .* ...
    sum([data.Speed_J0,data.Speed_J1,data.Speed_J2,...
         data.Speed_J3,data.Speed_J4,data.Speed_J5],2);

data.StressDelta = [NaN; diff(data.MechanicalStress)];

%% =======================
% 6. REMOVE NaNs
% ========================

data = rmmissing(data);

%% =======================
% 7. PREPARE FEATURES
% ========================

predictorVars = data.Properties.VariableNames;

predictorVars = setdiff(predictorVars, { ...
    'Robot_ProtectiveStop','grip_lost','Failure',...
    'Timestamp','cycle','Num'});

X = data{:, predictorVars};
Y = data.Failure;

%% =======================
% 8. TIME SPLIT
% ========================

n = height(data);
splitIdx = floor(0.7 * n);

X_train = X(1:splitIdx,:);
Y_train = Y(1:splitIdx);

X_test = X(splitIdx+1:end,:);
Y_test = Y(splitIdx+1:end);

%% =======================
% 9. INITIAL MODEL (FOR FEATURE IMPORTANCE)
% ========================

model = TreeBagger(100, X_train, Y_train, ...
    'Method','classification', ...
    'OOBPrediction','On', ...
    'OOBPredictorImportance','on');

%% =======================
% 10. FEATURE PRUNING
% ========================

importance = model.OOBPermutedPredictorDeltaError;
normImp = importance / max(importance);

keepIdx = normImp > 0.3;

prunedVars = predictorVars(keepIdx);

X_train_pruned = X_train(:, keepIdx);
X_test_pruned  = X_test(:, keepIdx);

fprintf('Original features: %d | Kept: %d\n', ...
    length(predictorVars), sum(keepIdx));

%% =======================
% 11. BALANCE TRAINING DATA (CRITICAL)
% ========================

idxFail = find(Y_train == 1);
idxNorm = find(Y_train == 0);

% Keep all failures, undersample normals
idxNormSample = randsample(idxNorm, min(length(idxNorm), 2*length(idxFail)));

idxFinal = [idxFail; idxNormSample];

X_train_bal = X_train_pruned(idxFinal,:);
Y_train_bal = Y_train(idxFinal);


%% =======================
% 12. TRAIN FINAL MODEL
% ========================

model_pruned = TreeBagger(500, X_train_bal, Y_train_bal, ...
    'Method','classification', ...
    'OOBPrediction','On', ...
    'OOBPredictorImportance','on');   % <-- ADD THIS

%% =======================
% 13. PROBABILITY PREDICTION
% ========================

[~, score] = predict(model_pruned, X_test_pruned);

failProb = score(:,2);

%% =======================
% 14. THRESHOLD 
% ========================

threshold = 0.3;   % 

Y_pred = failProb > threshold;

%% =======================
% 15. CONFUSION MATRIX
% ========================

figure;

Y_test_lbl = categorical(Y_test,[0 1],{'No Failure','Failure'});
Y_pred_lbl = categorical(Y_pred,[0 1],{'No Failure','Failure'});

confusionchart(Y_test_lbl, Y_pred_lbl);


%% =======================
% 16. METRICS 
% ========================

TP = sum((Y_pred==1) & (Y_test==1));
FP = sum((Y_pred==1) & (Y_test==0));
FN = sum((Y_pred==0) & (Y_test==1));

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);

fprintf('\nPrecision: %.3f\n', precision);
fprintf('Recall (FAILURE DETECTION): %.3f\n', recall);
fprintf('Predicted failure rate: %.3f\n', mean(Y_pred));

%% =======================
% 17. FEATURE IMPORTANCE PLOT
% ========================

importance = model_pruned.OOBPermutedPredictorDeltaError;

% Sort descending
[sortedImp, idx] = sort(importance, 'descend');
sortedNames = prunedVars(idx);

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

    % -----------------------
    % Base naming
    % -----------------------
    % -----------------------
    % Base naming (FIXED ORDER)
    % -----------------------

    % ---- ENGINEERED FEATURES FIRST ----
    if strcmp(name, 'TotalCurrent')
    base = "Total Current";

    elseif strcmp(name, 'CurrentImbalance')
    base = "Current Imbalance";

    elseif strcmp(name, 'TotalTemp')
    base = "Total Temp";

    elseif strcmp(name, 'MechanicalStress')
    base = "Mechanical Stress";

    elseif strcmp(name, 'StressDelta')
    base = "Stress Δ";

    % ---- THEN SENSOR FEATURES ----
    elseif contains(name, 'Temperature')
    base = "Temp J" + jointID;

    elseif contains(name, 'Current') && ~contains(name,'Total')
    base = "Current J" + jointID;

    elseif contains(name, 'Speed')
    base = "Speed J" + jointID;

    elseif contains(name, 'Tool_current')
    base = "Tool Current";

    else
    base = name;
    end

    % -----------------------
    % Trend suffix
    % -----------------------
    if contains(name, 'delta')
        base = base + " Δ";
    elseif contains(name, 'rollmean')
        base = base + " R.M.";
    end

    cleanNames(i) = base;
end

% -----------------------
% Assign colors
% -----------------------

colors = lines(7); % J0–J5
toolColor = [0 0 0];

barColors = zeros(length(sortedNames),3);

for i = 1:length(sortedNames)
    name = sortedNames{i};

    if contains(name, 'Tool_current')
        barColors(i,:) = toolColor;

    % joint-based coloring
    elseif ~isempty(regexp(name, '(J|T)(\d)', 'once'))
        tokens = regexp(name, '(J|T)(\d)', 'tokens');
        j = str2double(tokens{1}{2}) + 1;
        barColors(i,:) = colors(j,:);

    % engineered features = gray
    else
        barColors(i,:) = [0.4 0.4 0.4];
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

title('Feature Importance (Pruned Model)');
xlabel('Importance Score');
set(gca, 'FontSize', 11);

% -----------------------
% LEGEND
% -----------------------

hold on;

h = gobjects(10,1);

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

% Engineered features
h(8) = plot(nan,nan,'s', ...
    'MarkerFaceColor',[0.4 0.4 0.4], ...
    'MarkerEdgeColor',[0.4 0.4 0.4], ...
    'MarkerSize',8);

% Δ and R.M.
h(9) = plot(nan,nan,'k-','LineWidth',1.5);
h(10)= plot(nan,nan,'k--','LineWidth',1.5);

legendLabels = [
    jointLabels, ...
    {'Engineered Features','Δ = change','R.M. = rolling mean'}
];

legend(h, legendLabels, 'Location','best');