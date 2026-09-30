%% IS LAB 3 - PAPILDOMA UZDUOTIS
% RBF tinklo mokymas, kai automatiskai atnaujinami:
%   w1, w2, w0
%   c1, c2
%   r1, r2
%
% Naudojamas gradientinis mokymas.
%
% F = exp(-(x-c)^2/(2*r^2))
% y = w1*F1 + w2*F2 + w0
% e = d-y

clear;
clc;
close all;

%% 1. Mokymo duomenys

x = 0.1:1/22:1;

d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;

%% 2. Pradines RBF parametru reiksmes
% Sios reiksmes dabar tik PRADINES - mokymo metu jos keisis.

c1 = 0.25;
c2 = 0.75;

r1 = 0.22;
r2 = 0.22;

%% 3. Pradiniai isejimo svoriai

rng(1);

w1 = 0.4*rand(1)-0.2;
w2 = 0.4*rand(1)-0.2;
w0 = 0.4*rand(1)-0.2;

%% 4. Mokymosi greiciai

eta_w = 0.05;       % svoriams
eta_c = 0.001;      % centrams
eta_r = 0.001;      % spinduliams

MSE = 1;
epoch = 0;

maxEpochs = 20000;
MSE_goal = 0.0001;

%% 5. Mokymas

while MSE > MSE_goal && epoch < maxEpochs

    epoch = epoch + 1;
    error_sum = 0;

    for i = 1:length(x)

        %% RBF funkcijos

        F1 = exp(-(x(i)-c1)^2/(2*r1^2));
        F2 = exp(-(x(i)-c2)^2/(2*r2^2));

        %% Tinklo isejimas

        y = w1*F1 + w2*F2 + w0;

        %% Klaida

        e = d(i) - y;

        error_sum = error_sum + e^2;

        %% Issaugome senus svorius gradientams
        % Centro ir spindulio gradientai turi buti skaiciuojami
        % su svoriais, kurie buvo naudojami skaiciuojant y.

        w1_old = w1;
        w2_old = w2;

        %% 1 RBF centro ir spindulio korekcijos

        delta_c1 = eta_c * e * w1_old * F1 * ...
                   (x(i)-c1)/(r1^2);

        delta_r1 = eta_r * e * w1_old * F1 * ...
                   ((x(i)-c1)^2)/(r1^3);

        %% 2 RBF centro ir spindulio korekcijos

        delta_c2 = eta_c * e * w2_old * F2 * ...
                   (x(i)-c2)/(r2^2);

        delta_r2 = eta_r * e * w2_old * F2 * ...
                   ((x(i)-c2)^2)/(r2^3);

        %% Isejimo svoriu atnaujinimas

        w1 = w1 + eta_w*e*F1;
        w2 = w2 + eta_w*e*F2;
        w0 = w0 + eta_w*e;

        %% Centru atnaujinimas

        c1 = c1 + delta_c1;
        c2 = c2 + delta_c2;

        %% Spinduliu atnaujinimas

        r1 = r1 + delta_r1;
        r2 = r2 + delta_r2;

        % Spindulys negali tapti 0 ar neigiamas
        if r1 < 0.03
            r1 = 0.03;
        end

        if r2 < 0.03
            r2 = 0.03;
        end

    end

    MSE = error_sum/length(x);
    MSE_history(epoch) = MSE;

end

%% 6. Rezultatai

fprintf('--- IS LAB 3 PAPILDOMA UZDUOTIS ---\n');
fprintf('Epochu skaicius = %d\n', epoch);
fprintf('MSE = %.8f\n\n', MSE);

fprintf('Automatiskai ismokti RBF parametrai:\n');
fprintf('c1 = %.6f\n', c1);
fprintf('r1 = %.6f\n', r1);
fprintf('c2 = %.6f\n', c2);
fprintf('r2 = %.6f\n\n', r2);

fprintf('Ismokti isejimo parametrai:\n');
fprintf('w1 = %.6f\n', w1);
fprintf('w2 = %.6f\n', w2);
fprintf('w0 = %.6f\n', w0);

%% 7. Patikrinimas

x_test = 0:0.002:1;

for i = 1:length(x_test)

    F1_test(i) = exp(-(x_test(i)-c1)^2/(2*r1^2));
    F2_test(i) = exp(-(x_test(i)-c2)^2/(2*r2^2));

    y_test(i) = w1*F1_test(i) + ...
                w2*F2_test(i) + w0;

end

d_test = (1 + 0.6*sin(2*pi*x_test/0.7) + ...
              0.3*sin(2*pi*x_test))/2;

%% 8. Aproksimavimo grafikas

figure;

plot(x_test,d_test,'--','LineWidth',1.5);
hold on;
plot(x_test,y_test,'LineWidth',1.5);
plot(x,d,'o','MarkerSize',6);

grid on;

xlabel('x');
ylabel('y');

legend('Norima funkcija', ...
       'RBF tinklo atsakas', ...
       'Mokymo taskai', ...
       'Location','best');

title('RBF aproksimacija mokant c, r ir w');

%% 9. Ismoktos RBF funkcijos

figure;

plot(x_test,F1_test,'LineWidth',1.5);
hold on;
plot(x_test,F2_test,'LineWidth',1.5);

grid on;

xlabel('x');
ylabel('F');

legend('F_1','F_2');
title('Automatiskai ismoktos RBF funkcijos');

%% 10. MSE grafikas

figure;

plot(1:length(MSE_history),MSE_history,'LineWidth',1.5);

grid on;

xlabel('Epocha');
ylabel('MSE');

title('Papildomos uzduoties mokymo paklaida');
