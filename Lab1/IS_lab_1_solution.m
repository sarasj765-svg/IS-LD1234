%% IS Lab 1 - Perceptron + Naive Bayes
% Pagal https://github.com/serackis/IS-Lab1
% 1) Perceptrono klasifikatorius pagal 2 pozymius
% 2) Perceptrono mokymo algoritmas
% 3) Papildoma uzduotis: Naive Bayes klasifikatorius

clear;
clc;
close all;

% Reikalavime nurodyta pradines reiksmes generuoti atsitiktinai
rng('shuffle');

%% Vaizdu nuskaitymas
A1 = imread('apple_04.jpg');
A2 = imread('apple_05.jpg');
A3 = imread('apple_06.jpg');
A4 = imread('apple_07.jpg');
A5 = imread('apple_11.jpg');
A6 = imread('apple_12.jpg');
A7 = imread('apple_13.jpg');
A8 = imread('apple_17.jpg');
A9 = imread('apple_19.jpg');

P1 = imread('pear_01.jpg');
P2 = imread('pear_02.jpg');
P3 = imread('pear_03.jpg');
P4 = imread('pear_09.jpg');

%% Pozymiu skaiciavimas: spalva ir apvalumas
hsv_value_A1 = spalva_color(A1); metric_A1 = apvalumas_roundness(A1);
hsv_value_A2 = spalva_color(A2); metric_A2 = apvalumas_roundness(A2);
hsv_value_A3 = spalva_color(A3); metric_A3 = apvalumas_roundness(A3);
hsv_value_A4 = spalva_color(A4); metric_A4 = apvalumas_roundness(A4);
hsv_value_A5 = spalva_color(A5); metric_A5 = apvalumas_roundness(A5);
hsv_value_A6 = spalva_color(A6); metric_A6 = apvalumas_roundness(A6);
hsv_value_A7 = spalva_color(A7); metric_A7 = apvalumas_roundness(A7);
hsv_value_A8 = spalva_color(A8); metric_A8 = apvalumas_roundness(A8);
hsv_value_A9 = spalva_color(A9); metric_A9 = apvalumas_roundness(A9);

hsv_value_P1 = spalva_color(P1); metric_P1 = apvalumas_roundness(P1);
hsv_value_P2 = spalva_color(P2); metric_P2 = apvalumas_roundness(P2);
hsv_value_P3 = spalva_color(P3); metric_P3 = apvalumas_roundness(P3);
hsv_value_P4 = spalva_color(P4); metric_P4 = apvalumas_roundness(P4);

%% ============================================================
%  1 UZDUOTIS - PERCEPTRONO ISEJIMO SKAICIAVIMAS
%  Mokymui pagal template naudojami A1, A2, A3, P1, P2
%% ============================================================

x1 = [hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2 = [metric_A1    metric_A2    metric_A3    metric_P1    metric_P2];

P = [x1; x2];

% Norimi atsakymai: obuolys = +1, kriause = -1
T = [1; 1; 1; -1; -1];

% Atsitiktines pradines perceptrono parametro reiksmes
w1 = randn(1);
w2 = randn(1);
b  = randn(1);

fprintf('Pradines reiksmes:\n');
fprintf('w1 = %.6f, w2 = %.6f, b = %.6f\n\n', w1, w2, b);

% Apskaiciuojame pradines klasifikatoriaus isvestis visiems 5 pavyzdziams
v = zeros(1,5);
y = zeros(1,5);
err = zeros(1,5);

for n = 1:5
    v(n) = x1(n)*w1 + x2(n)*w2 + b;

    if v(n) > 0
        y(n) = 1;
    else
        y(n) = -1;
    end

    err(n) = T(n) - y(n);
end

e = sum(abs(err));

fprintf('Pradine perceptrono isvestis:\n');
disp(y);
fprintf('Norima isvestis:\n');
disp(T');
fprintf('Pradine bendra paklaida e = %.0f\n\n', e);

%% ============================================================
%  2 UZDUOTIS - PERCEPTRONO MOKYMAS
%% ============================================================

eta = 0.1;          % 0 < eta < 1
epoch = 0;
maxEpochs = 100000; % tik apsauga nuo begalinio ciklo, jei kode butu klaida
error_history = [];

while e ~= 0 && epoch < maxEpochs
    epoch = epoch + 1;

    % Online perceptrono mokymas - kiekvienas pavyzdys is eiles
    for n = 1:5
        % Dabartine svertine suma
        v_current = x1(n)*w1 + x2(n)*w2 + b;

        % Dabartinis perceptrono atsakymas
        if v_current > 0
            y_current = 1;
        else
            y_current = -1;
        end

        % Momentine klaida: e(n) = d(n) - y(n)
        e_current = T(n) - y_current;

        % Parametru atnaujinimas tiksliai pagal uzduoties formules
        w1 = w1 + eta * e_current * x1(n);
        w2 = w2 + eta * e_current * x2(n);
        b  = b  + eta * e_current;
    end

    % Po epochos patikriname visus 5 mokymo pavyzdzius
    for n = 1:5
        v(n) = x1(n)*w1 + x2(n)*w2 + b;

        if v(n) > 0
            y(n) = 1;
        else
            y(n) = -1;
        end

        err(n) = T(n) - y(n);
    end

    e = sum(abs(err));
    error_history(epoch) = e;
end

if e == 0
    fprintf('Mokymas baigtas sekmingai.\n');
    fprintf('Epochu skaicius: %d\n', epoch);
else
    error('Perceptronas nepasieke e = 0 per %d epochu.', maxEpochs);
end

fprintf('Galutiniai parametrai:\n');
fprintf('w1 = %.6f\n', w1);
fprintf('w2 = %.6f\n', w2);
fprintf('b  = %.6f\n\n', b);

fprintf('Galutine klasifikacija mokymo rinkiniui:\n');
disp(y);
fprintf('Norima klasifikacija:\n');
disp(T');
fprintf('Galutine bendra paklaida e = %.0f\n\n', e);

%% Visu 13 paveiksleliu pozymiai - papildomam patikrinimui ir Naive Bayes

x1_all = [ ...
    hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 hsv_value_A5 ...
    hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 ...
    hsv_value_P1 hsv_value_P2 hsv_value_P3 hsv_value_P4 ];

x2_all = [ ...
    metric_A1 metric_A2 metric_A3 metric_A4 metric_A5 ...
    metric_A6 metric_A7 metric_A8 metric_A9 ...
    metric_P1 metric_P2 metric_P3 metric_P4 ];

T_all = [ones(9,1); -ones(4,1)];

%% Perceptrono patikrinimas su visais 13 pateiktu objektu
y_all_perceptron = zeros(13,1);

for n = 1:13
    v_all = x1_all(n)*w1 + x2_all(n)*w2 + b;

    if v_all > 0
        y_all_perceptron(n) = 1;
    else
        y_all_perceptron(n) = -1;
    end
end

perceptron_accuracy = mean(y_all_perceptron == T_all) * 100;

fprintf('Perceptrono klasifikacija visiems 13 objektu:\n');
disp([T_all y_all_perceptron]);
fprintf('Perceptrono tikslumas visuose 13 objektu: %.2f %%\n\n', perceptron_accuracy);

%% Perceptrono sprendimo ribos grafikas
figure;
hold on;
grid on;

scatter(x1_all(1:9),  x2_all(1:9),  70, 'o', 'filled');
scatter(x1_all(10:13), x2_all(10:13), 70, 's', 'filled');

xmin = min(x1_all) - 0.03;
xmax = max(x1_all) + 0.03;

if abs(w2) > eps
    xx = linspace(xmin, xmax, 200);
    yy = -(w1*xx + b) / w2;
    plot(xx, yy, 'LineWidth', 1.5);
end

xlabel('x_1 - spalva (HSV Hue)');
ylabel('x_2 - apvalumas');
title('Perceptrono klasifikavimas');
legend('Obuoliai (+1)', 'Kriauses (-1)', 'Sprendimo riba', 'Location', 'best');
hold off;

%% Mokymo paklaidos grafikas
if ~isempty(error_history)
    figure;
    plot(1:length(error_history), error_history, 'LineWidth', 1.5);
    grid on;
    xlabel('Epocha');
    ylabel('Bendra paklaida e');
    title('Perceptrono mokymo paklaida');
end

%% ============================================================
%  PAPILDOMA UZDUOTIS - NAIVE BAYES CLASSIFIER
%  Rankinis Gaussian Naive Bayes, kad nereiketu Statistics Toolbox.
%% ============================================================

X_all = [x1_all' x2_all'];  % 13x2
classes = [1 -1];

mu = zeros(2,2);
variance = zeros(2,2);
prior = zeros(1,2);

for c = 1:2
    idx = (T_all == classes(c));

    mu(c,:) = mean(X_all(idx,:), 1);
    variance(c,:) = var(X_all(idx,:), 0, 1);
    prior(c) = sum(idx) / length(T_all);

    % Apsauga nuo nulines dispersijos
    variance(c, variance(c,:) < 1e-12) = 1e-12;
end

logPosterior = zeros(size(X_all,1), 2);

for n = 1:size(X_all,1)
    for c = 1:2
        % Gaussian log tikimybe kiekvienam pozymiui
        logLikelihood = -0.5*log(2*pi*variance(c,:)) ...
            - ((X_all(n,:) - mu(c,:)).^2) ./ (2*variance(c,:));

        % Naive Bayes: pozymiu log-tikimybiu suma + log(prior)
        logPosterior(n,c) = sum(logLikelihood) + log(prior(c));
    end
end

[~, classIndex] = max(logPosterior, [], 2);
y_nb = classes(classIndex)';

nb_accuracy = mean(y_nb == T_all) * 100;

fprintf('Naive Bayes rezultatai [norima klase, prognozuota klase]:\n');
disp([T_all y_nb]);
fprintf('Naive Bayes tikslumas pateiktame 13 objektu rinkinyje: %.2f %%\n', nb_accuracy);
