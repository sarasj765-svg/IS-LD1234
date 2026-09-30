%% IS LAB 3 - PAGRINDINE UZDUOTIS
% RBF / SBF tinklas funkcijos aproksimavimui
%
% Struktura:
%   x -> 2 Gauso radialines bazines funkcijos -> tiesinis isejimas y
%
% F = exp(-(x-c)^2/(2*r^2))
%
% c1, c2, r1, r2 parenkami RANKINIU BUDU.
% Mokomi tik isejimo sluoksnio parametrai:
% w1, w2, w0.
%
% Atnaujinimas toks pats principas kaip perceptrono/LMS:
% w_new = w_old + eta*e*input

clear;
clc;
close all;

%% 1. Mokymo duomenys

x = 0.1:1/22:1;

% Norimas atsakas d
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;

%% 2. RBF parametrai parenkami rankiniu budu

% 1 RBF centras ir spindulys
c1 = 0.20;
r1 = 0.18;

% 2 RBF centras ir spindulys
c2 = 0.90;
r2 = 0.18;

%% 3. Pradines isejimo svoriu reiksmes

w1 = rand(1);
w2 = rand(1);
w0 = rand(1);

%% 4. Mokymo parametrai

eta = 0.05;
MSE = 1;
epoch = 0;
maxEpochs = 10000;
MSE_goal = 0.0002;

%% 5. Mokymas

while MSE > MSE_goal && epoch < maxEpochs

    epoch = epoch + 1;
    error_sum = 0;

    for i = 1:length(x)

        %% Gauso radialines bazines funkcijos

        F1 = exp(-(x(i)-c1)^2/(2*r1^2));
        F2 = exp(-(x(i)-c2)^2/(2*r2^2));

        %% Tiesinis isejimo neuronas

        y = F1*w1 + F2*w2 + w0;

        %% Klaida

        e = d(i) - y;

        error_sum = error_sum + e^2;

        %% Isejimo parametru atnaujinimas

        w1 = w1 + eta*e*F1;
        w2 = w2 + eta*e*F2;
        w0 = w0 + eta*e;

    end

    %% Vidutine kvadratine paklaida

    MSE = error_sum/length(x);
    MSE_history(epoch) = MSE;

end

%% 6. Rezultatai

fprintf('--- IS LAB 3 PAGRINDINE UZDUOTIS ---\n');
fprintf('Epochu skaicius = %d\n', epoch);
fprintf('MSE = %.8f\n\n', MSE);

fprintf('Rankiniu budu parinkti RBF parametrai:\n');
fprintf('c1 = %.4f, r1 = %.4f\n', c1, r1);
fprintf('c2 = %.4f, r2 = %.4f\n\n', c2, r2);

fprintf('Ismokti isejimo parametrai:\n');
fprintf('w1 = %.6f\n', w1);
fprintf('w2 = %.6f\n', w2);
fprintf('w0 = %.6f\n', w0);

%% 7. Tinklo patikrinimas tankiame tasku tinklelyje

x_test = 0:0.002:1;

for i = 1:length(x_test)

    F1_test(i) = exp(-(x_test(i)-c1)^2/(2*r1^2));
    F2_test(i) = exp(-(x_test(i)-c2)^2/(2*r2^2));

    y_test(i) = F1_test(i)*w1 + F2_test(i)*w2 + w0;

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

title('RBF funkcijos aproksimacija');

%% 9. RBF funkciju grafikas

figure;

plot(x_test,F1_test,'LineWidth',1.5);
hold on;
plot(x_test,F2_test,'LineWidth',1.5);

grid on;

xlabel('x');
ylabel('F');

legend('F_1','F_2');
title('Dvi Gauso radialines bazines funkcijos');

%% 10. Mokymo paklaidos grafikas

figure;

plot(1:length(MSE_history),MSE_history,'LineWidth',1.5);

grid on;

xlabel('Epocha');
ylabel('MSE');

title('RBF tinklo mokymo paklaida');
