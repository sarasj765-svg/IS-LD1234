%% IS LAB 4 - PAPILDOMA UZDUOTIS
% Ta pati ranka rasytu skaitmenu atpazinimo uzduotis,
% bet naudojamas daugiasluoksnis perceptronas (MLP).
%
% Reikalingi:
%   train_digits.png
%   test_digits.png
%   extract_symbol_features.m
%
% MATLAB Deep Learning Toolbox reikalingas feedforwardnet funkcijai.

clear;
clc;
close all;

%% 1. Nustatymai

trainImage = 'train_digits.png';
testImage  = 'test_digits.png';

numberOfTrainingRows = 8;
numberOfTestRows = 1;

symbols = '0123456789';

% Pvz. '2026'. Jei nenori skaiciuoti accuracy - palik ''.
expectedTest = '';

%% 2. Pozymiai

P = extract_symbol_features(trainImage,numberOfTrainingRows,true);

numberOfClasses = length(symbols);

expectedTrainingSymbols = ...
    numberOfClasses*numberOfTrainingRows;

if size(P,2) ~= expectedTrainingSymbols
    error(['Mokymo paveikslelyje turetu buti %d simboliu, ' ...
           'bet isskirta %d.'], ...
           expectedTrainingSymbols,size(P,2));
end

%% 3. Teisingi atsakymai

T = repmat(eye(numberOfClasses), ...
           1,numberOfTrainingRows);

%% ============================================================
% 4. MLP TINKLAS
% ============================================================

% Paslepto sluoksnio neuronu skaicius
hiddenNeurons = 12;

netMLP = feedforwardnet(hiddenNeurons,'trainscg');

% Naudojame visa musu parengta mokymo imti mokymui.
netMLP.divideFcn = 'dividetrain';

% Pasleptas sluoksnis - netiesinis
netMLP.layers{1}.transferFcn = 'tansig';

% Isejimo sluoksnis paliekamas tiesinis (purelin)
netMLP.layers{2}.transferFcn = 'purelin';

netMLP.performFcn = 'mse';

netMLP.trainParam.epochs = 1000;
netMLP.trainParam.goal = 1e-4;

%% 5. MLP mokymas

fprintf('\nMokomas MLP tinklas...\n');

netMLP = train(netMLP,P,T);

%% 6. Mokymo rinkinio accuracy

Ytrain = sim(netMLP,P);

[~,predictedTrain] = max(Ytrain,[],1);
[~,correctTrain] = max(T,[],1);

trainingAccuracy = ...
    mean(predictedTrain == correctTrain)*100;

fprintf('\nMLP mokymo accuracy = %.2f %%\n', ...
        trainingAccuracy);

%% ============================================================
% 7. Testavimo paveikslelis
% ============================================================

Ptest = extract_symbol_features(testImage,numberOfTestRows,true);

Ytest = sim(netMLP,Ptest);

[~,predictedTest] = max(Ytest,[],1);

answerMLP = symbols(predictedTest);

fprintf('\n--- MLP TESTAVIMO REZULTATAS ---\n');
fprintf('%s\n',answerMLP);

%% 8. Test accuracy, jei zinomas tikras atsakymas

if ~isempty(expectedTest)

    if length(expectedTest) ~= length(answerMLP)

        warning(['expectedTest ilgis nesutampa su ' ...
                 'atpazintu simboliu skaiciumi.']);

    else

        testAccuracy = ...
            mean(answerMLP == expectedTest)*100;

        fprintf('Tikras atsakymas: %s\n',expectedTest);
        fprintf('MLP test accuracy = %.2f %%\n', ...
                testAccuracy);

    end

end

%% 9. Rezultato atvaizdavimas

figure;

text(0.05,0.5, ...
     ['MLP: ',answerMLP], ...
     'FontSize',26);

axis off;

title('MLP atpazinimo rezultatas');
