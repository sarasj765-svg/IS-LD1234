%% IS-Lab2 PAPILDOMA UZDUOTIS
% Pavirsiaus aproksimavimas:
% 2 iejimai -> 4 paslepti tanh neuronai -> 1 tiesinis isejimas
% Mokymas - Backpropagation
%
% Pagal destytojo rasymo stiliu:
% visi neuronai, svoriai ir delta reiksmes parasytos atskirai.

clear;
clc;
close all;

%% 1. Mokymo duomenys

% Du iejimai x1 ir x2
x1_values = 0:0.1:1;
x2_values = 0:0.1:1;

% Susikuriame visas x1 ir x2 kombinacijas
k = 1;

for i = 1:length(x1_values)

    for j = 1:length(x2_values)

        x1(k) = x1_values(i);
        x2(k) = x2_values(j);

        % Norimas atsakas - pavirsiaus funkcija
        % Ji naudojama tik mokymo atsakui d sudaryti.
        d(k) = (sin(pi*x1(k))*cos(pi*x2(k)) + 1)/2;

        k = k + 1;

    end

end

%% 2. Pradines svoriu ir bias reiksmes

rng(1);

% ============================================================
% 1 sluoksnis
% Kiekvienas pasleptas neuronas turi DU iejimo svorius:
% viena svori x1 ir viena svori x2
% ============================================================

% Neuronas 1
w11_1 = rand(1);
w12_1 = rand(1);
b1_1  = rand(1);

% Neuronas 2
w21_1 = rand(1);
w22_1 = rand(1);
b2_1  = rand(1);

% Neuronas 3
w31_1 = rand(1);
w32_1 = rand(1);
b3_1  = rand(1);

% Neuronas 4
w41_1 = rand(1);
w42_1 = rand(1);
b4_1  = rand(1);

% ============================================================
% 2 sluoksnis
% 4 paslepti neuronai -> 1 isejimas
% ============================================================

w11_2 = rand(1);
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);

b1_2 = rand(1);

%% 3. Mokymosi greitis

eta = 0.1;

%% 4. Mokymo nustatymai

epoch = 0;
MSE = 1;

%% 5. MOKYMAS

while MSE > 0.002 && epoch < 50000

    epoch = epoch + 1;

    error_sum = 0;

    for i = 1:length(x1)

        %% ====================================================
        % 1 sluoksnio pasvertos sumos
        % Dabar kiekvienas neuronas gauna x1 IR x2
        % =====================================================

        v1_1 = x1(i)*w11_1 + x2(i)*w12_1 + b1_1;
        v2_1 = x1(i)*w21_1 + x2(i)*w22_1 + b2_1;
        v3_1 = x1(i)*w31_1 + x2(i)*w32_1 + b3_1;
        v4_1 = x1(i)*w41_1 + x2(i)*w42_1 + b4_1;

        %% 1 sluoksnio aktyvacijos funkcija

        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);

        %% ====================================================
        % 2 sluoksnio pasverta suma
        % =====================================================

        v1_2 = y1_1*w11_2 + ...
               y2_1*w12_2 + ...
               y3_1*w13_2 + ...
               y4_1*w14_2 + b1_2;

        %% Isejimo sluoksnio aktyvacija
        % Tiesine aktyvacija

        y1_2 = v1_2;

        %% Tinklo isejimas

        y = y1_2;

        %% Klaida

        e = d(i) - y;

        error_sum = error_sum + e^2;

        %% ====================================================
        % BACKPROPAGATION
        % =====================================================

        %% Isejimo sluoksnio delta
        % Tiesines aktyvacijos isvestine = 1

        delta1_2 = e;

        %% Pasleptojo sluoksnio delta
        % tanh isvestine = 1 - y^2

        delta1_1 = (1-y1_1^2)*delta1_2*w11_2;
        delta2_1 = (1-y2_1^2)*delta1_2*w12_2;
        delta3_1 = (1-y3_1^2)*delta1_2*w13_2;
        delta4_1 = (1-y4_1^2)*delta1_2*w14_2;

        %% ====================================================
        % Isejimo sluoksnio svoriu atnaujinimas
        % =====================================================

        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w12_2 = w12_2 + eta*delta1_2*y2_1;
        w13_2 = w13_2 + eta*delta1_2*y3_1;
        w14_2 = w14_2 + eta*delta1_2*y4_1;

        b1_2 = b1_2 + eta*delta1_2;

        %% ====================================================
        % Pasleptojo sluoksnio svoriu atnaujinimas
        %
        % Kiekvienam neuronui reikia atnaujinti:
        % - svori nuo x1
        % - svori nuo x2
        % =====================================================

        % Neuronas 1
        w11_1 = w11_1 + eta*delta1_1*x1(i);
        w12_1 = w12_1 + eta*delta1_1*x2(i);

        % Neuronas 2
        w21_1 = w21_1 + eta*delta2_1*x1(i);
        w22_1 = w22_1 + eta*delta2_1*x2(i);

        % Neuronas 3
        w31_1 = w31_1 + eta*delta3_1*x1(i);
        w32_1 = w32_1 + eta*delta3_1*x2(i);

        % Neuronas 4
        w41_1 = w41_1 + eta*delta4_1*x1(i);
        w42_1 = w42_1 + eta*delta4_1*x2(i);

        %% Bias atnaujinimas

        b1_1 = b1_1 + eta*delta1_1;
        b2_1 = b2_1 + eta*delta2_1;
        b3_1 = b3_1 + eta*delta3_1;
        b4_1 = b4_1 + eta*delta4_1;

    end

    %% Vidutine kvadratine paklaida

    MSE = error_sum/length(x1);

    MSE_history(epoch) = MSE;

end

%% 6. Rezultatai

fprintf('Papildoma uzduotis baigta\n');
fprintf('Epochu skaicius = %d\n',epoch);
fprintf('MSE = %.6f\n',MSE);

%% ============================================================
% 7. TINKLO PATIKRINIMAS IR 3D PAVIRSIAUS BRAIZYMAS
% ============================================================

x1_test_values = 0:0.025:1;
x2_test_values = 0:0.025:1;

[X1_test,X2_test] = meshgrid(x1_test_values,x2_test_values);

Y_test = zeros(size(X1_test));

for i = 1:size(X1_test,1)

    for j = 1:size(X1_test,2)

        x1_t = X1_test(i,j);
        x2_t = X2_test(i,j);

        %% 1 sluoksnis

        v1_1 = x1_t*w11_1 + x2_t*w12_1 + b1_1;
        v2_1 = x1_t*w21_1 + x2_t*w22_1 + b2_1;
        v3_1 = x1_t*w31_1 + x2_t*w32_1 + b3_1;
        v4_1 = x1_t*w41_1 + x2_t*w42_1 + b4_1;

        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);

        %% 2 sluoksnis

        v1_2 = y1_1*w11_2 + ...
               y2_1*w12_2 + ...
               y3_1*w13_2 + ...
               y4_1*w14_2 + b1_2;

        %% Tinklo prognoze

        Y_test(i,j) = v1_2;

    end

end

%% Tikrasis pavirsius palyginimui

D_test = (sin(pi*X1_test).*cos(pi*X2_test) + 1)/2;

%% 8. Tikrasis pavirsius

figure;

surf(X1_test,X2_test,D_test);

grid on;

xlabel('x1');
ylabel('x2');
zlabel('d');

title('Tikrasis pavirsius');

%% 9. Neuroninio tinklo aproksimuotas pavirsius

figure;

surf(X1_test,X2_test,Y_test);

grid on;

xlabel('x1');
ylabel('x2');
zlabel('y');

title('Neuroninio tinklo aproksimuotas pavirsius');

%% 10. Abu pavirsiai viename grafike

figure;

surf(X1_test,X2_test,D_test);
hold on;

mesh(X1_test,X2_test,Y_test);

grid on;

xlabel('x1');
ylabel('x2');
zlabel('y');

title('Tikrasis ir MLP aproksimuotas pavirsiai');

legend('Tikrasis pavirsius','MLP aproksimacija');

%% 11. Mokymo paklaidos grafikas

figure;

plot(1:length(MSE_history),MSE_history,'LineWidth',1.5);

grid on;

xlabel('Epocha');
ylabel('MSE');

title('Papildomos uzduoties mokymo paklaida');
