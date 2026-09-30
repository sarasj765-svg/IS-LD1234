%% IS LAB 4 - PAGRINDINE UZDUOTIS
% Ranka rasytu SKAITMENU atpazinimas naudojant RBF tinkla.
%
% Reikia savo dvieju paveiksleliu:
%   train_digits.png
%   test_digits.png
%
% train_digits.png pavyzdys:
% kiekvienoje eiluteje ranka parasykite:
% 0 1 2 3 4 5 6 7 8 9
%
% Pakartokite ta pacia simboliu tvarka keliose eilutese.
%
% Uzduoties esme:
% 1) istraukti 35 pozymius,
% 2) sukurti RBF atpazintuva,
% 3) sumazinti RBF neuronu skaiciu nuo 13,
% 4) palyginti rezultatus.

clear;
clc;
close all;

%% ============================================================
% 1. NUSTATYMAI - PASIKEISK PAGAL SAVO PAVEIKSLELIUS
% ============================================================

trainImage = 'train_digits.png';
testImage  = 'test_digits.png';

% Kiek simboliu eiluciu yra mokymo paveikslelyje?
numberOfTrainingRows = 8;

% Kiek simboliu eiluciu yra testavimo paveikslelyje?
numberOfTestRows = 1;

% Simboliai PRIVALO buti tokia pacia tvarka kiekvienoje mokymo eiluteje.
symbols = '0123456789';

% Jei zinai, kas parasyta testavimo paveikslelyje, gali irasyti,
% pvz. expectedTest = '2026';
% Jei nenori skaiciuoti test accuracy - palik ''.
expectedTest = '';

%% 2. Pozymiai mokymui

P = extract_symbol_features(trainImage,numberOfTrainingRows,true);

numberOfClasses = length(symbols);
numberOfTrainingSymbols = size(P,2);

expectedTrainingSymbols = ...
    numberOfClasses*numberOfTrainingRows;

if numberOfTrainingSymbols ~= expectedTrainingSymbols
    error(['Mokymo paveikslelyje turetu buti %d simboliu, ' ...
           'bet isskirta %d.'], ...
           expectedTrainingSymbols,numberOfTrainingSymbols);
end

%% 3. Teisingu atsakymu matrica T

% Vienai eilutei:
% 0 -> [1 0 0 ...]'
% 1 -> [0 1 0 ...]'
% ...
% 9 -> [0 0 ... 1]'

T_oneRow = eye(numberOfClasses);

% Kadangi kiekviena mokymo eilute turi ta pacia simboliu tvarka,
% kartojame eye(...) tiek kartu, kiek yra eiluciu.
T = repmat(T_oneRow,1,numberOfTrainingRows);

%% ============================================================
% 4. RBF TINKLAS SU 13 NEURONU
% ============================================================

goal = 0;
spread = 1;

fprintf('\nMokomas RBF tinklas su max 13 neuronu...\n');

net13 = newrb(P,T,goal,spread,13);

Y13_train = sim(net13,P);

[~,class13_train] = max(Y13_train,[],1);
[~,correctTrain] = max(T,[],1);

accuracy13_train = ...
    mean(class13_train == correctTrain)*100;

%% ============================================================
% 5. SUMAZINTAS RBF TINKLAS
% ============================================================

% Uzduotyje reikia 13 sumazinti iki pasirinkto mazesnio skaiciaus.
reducedNeurons = 8;

fprintf('\nMokomas sumazintas RBF tinklas su max %d neuronais...\n', ...
        reducedNeurons);

netReduced = newrb(P,T,goal,spread,reducedNeurons);

YReduced_train = sim(netReduced,P);

[~,classReduced_train] = max(YReduced_train,[],1);

accuracyReduced_train = ...
    mean(classReduced_train == correctTrain)*100;

%% 6. Mokymo rezultatu palyginimas

fprintf('\n--- MOKYMO RINKINIO REZULTATAI ---\n');
fprintf('RBF max 13 neuronu accuracy = %.2f %%\n', ...
        accuracy13_train);

fprintf('RBF max %d neuronu accuracy = %.2f %%\n', ...
        reducedNeurons,accuracyReduced_train);

fprintf('\nFaktinis 13 tinklo RBF neuronu skaicius = %d\n', ...
        net13.layers{1}.size);

fprintf('Faktinis sumazinto tinklo RBF neuronu skaicius = %d\n', ...
        netReduced.layers{1}.size);

%% ============================================================
% 7. TESTAVIMO PAVEIKSLELIS
% ============================================================

Ptest = extract_symbol_features(testImage,numberOfTestRows,true);

%% 13 neuronu tinklo atsakymas

Y13_test = sim(net13,Ptest);

[~,idx13] = max(Y13_test,[],1);

answer13 = symbols(idx13);

%% Sumazinto tinklo atsakymas

YReduced_test = sim(netReduced,Ptest);

[~,idxReduced] = max(YReduced_test,[],1);

answerReduced = symbols(idxReduced);

fprintf('\n--- TESTAVIMO REZULTATAI ---\n');
fprintf('RBF max 13 neuronu: %s\n',answer13);
fprintf('RBF max %d neuronu: %s\n', ...
        reducedNeurons,answerReduced);

%% 8. Jei nurodytas teisingas testavimo atsakymas

if ~isempty(expectedTest)

    if length(expectedTest) ~= length(answer13)
        warning(['expectedTest ilgis nesutampa su atpazintu ' ...
                 'simboliu skaiciumi.']);
    else

        accuracy13_test = ...
            mean(answer13 == expectedTest)*100;

        accuracyReduced_test = ...
            mean(answerReduced == expectedTest)*100;

        fprintf('\nTikras atsakymas: %s\n',expectedTest);

        fprintf('13 neuronu test accuracy = %.2f %%\n', ...
                accuracy13_test);

        fprintf('%d neuronu test accuracy = %.2f %%\n', ...
                reducedNeurons,accuracyReduced_test);

    end

end

%% 9. Rezultatas lange

figure;

text(0.05,0.65, ...
     ['13 neuronu: ',answer13], ...
     'FontSize',22);

text(0.05,0.35, ...
     [num2str(reducedNeurons),' neuronu: ',answerReduced], ...
     'FontSize',22);

axis off;

title('RBF atpazinimo rezultatai');
