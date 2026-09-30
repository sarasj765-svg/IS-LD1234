%% IS-Lab2 - variantas pagal destytojo rasymo stiliu
% 1 iejimas -> 4 paslepti tanh neuronai -> 1 tiesinis isejimas
% Mokymas - Backpropagation

clear;
clc;
close all;

%% Mokymo duomenys
x = 0.1:1/22:1;

% Norimas atsakas
d = ((1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x))/2;

%% Pradines svoriu reiksmes

% 1 sluoksnis
w11_1 = rand(1);
w21_1 = rand(1);
w31_1 = rand(1);
w41_1 = rand(1);

b1_1 = rand(1);
b2_1 = rand(1);
b3_1 = rand(1);
b4_1 = rand(1);

% 2 sluoksnis
w11_2 = rand(1);
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);

b1_2 = rand(1);

%% Mokymosi greitis
eta = 0.1;

%% Mokymo nustatymai
epoch = 0;
MSE = 1;

%% Mokymas
while MSE > 0.001 && epoch < 50000

    epoch = epoch + 1;
    error_sum = 0;

    for i = 1:length(x)

        %% 1 sluoksnio pasvertos sumos
        v1_1 = x(i)*w11_1 + b1_1;
        v2_1 = x(i)*w21_1 + b2_1;
        v3_1 = x(i)*w31_1 + b3_1;
        v4_1 = x(i)*w41_1 + b4_1;

        %% 1 sluoksnio aktyvacijos funkcija
        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);

        %% 2 sluoksnio pasverta suma
        v1_2 = y1_1*w11_2 + ...
               y2_1*w12_2 + ...
               y3_1*w13_2 + ...
               y4_1*w14_2 + b1_2;

        %% Isejimo sluoksnio aktyvacija
        % Tiesine aktyvacijos funkcija
        y1_2 = v1_2;

        %% Tinklo isejimas
        y = y1_2;

        %% Klaida
        e = d(i) - y;

        error_sum = error_sum + e^2;

        %% Atnaujinami tinklo svoriai

        % Isejimo sluoksnio delta
        delta1_2 = e;

        % Pasleptojo sluoksnio delta
        % tanh isvestine = 1 - y^2
        delta1_1 = (1-y1_1^2)*delta1_2*w11_2;
        delta2_1 = (1-y2_1^2)*delta1_2*w12_2;
        delta3_1 = (1-y3_1^2)*delta1_2*w13_2;
        delta4_1 = (1-y4_1^2)*delta1_2*w14_2;

        %% Isejimo sluoksnio svoriai
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w12_2 = w12_2 + eta*delta1_2*y2_1;
        w13_2 = w13_2 + eta*delta1_2*y3_1;
        w14_2 = w14_2 + eta*delta1_2*y4_1;

        b1_2 = b1_2 + eta*delta1_2;

        %% Pasleptojo sluoksnio svoriai
        w11_1 = w11_1 + eta*delta1_1*x(i);
        w21_1 = w21_1 + eta*delta2_1*x(i);
        w31_1 = w31_1 + eta*delta3_1*x(i);
        w41_1 = w41_1 + eta*delta4_1*x(i);

        b1_1 = b1_1 + eta*delta1_1;
        b2_1 = b2_1 + eta*delta2_1;
        b3_1 = b3_1 + eta*delta3_1;
        b4_1 = b4_1 + eta*delta4_1;

    end

    %% Vidutine kvadratine paklaida
    MSE = error_sum/length(x);

    MSE_history(epoch) = MSE;

end

%% Rezultatai
fprintf('Epochu skaicius = %d\n', epoch);
fprintf('MSE = %.6f\n', MSE);

%% Tinklo isejimai mokymo taskuose
for i = 1:length(x)

    v1_1 = x(i)*w11_1 + b1_1;
    v2_1 = x(i)*w21_1 + b2_1;
    v3_1 = x(i)*w31_1 + b3_1;
    v4_1 = x(i)*w41_1 + b4_1;

    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);

    v1_2 = y1_1*w11_2 + ...
           y2_1*w12_2 + ...
           y3_1*w13_2 + ...
           y4_1*w14_2 + b1_2;

    y_mlp(i) = v1_2;

end

%% Tankesnis x vektorius grazesniam grafikui
% Cia tinklas jau NESIMOKO.
% Tiesiog patikriname, kokia lygia kreive jis ismoko.

x_test = 0:0.002:1;
y_test = zeros(size(x_test));

for i = 1:length(x_test)

    %% 1 sluoksnio pasvertos sumos
    v1_1 = x_test(i)*w11_1 + b1_1;
    v2_1 = x_test(i)*w21_1 + b2_1;
    v3_1 = x_test(i)*w31_1 + b3_1;
    v4_1 = x_test(i)*w41_1 + b4_1;

    %% 1 sluoksnio aktyvacijos funkcija
    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);

    %% 2 sluoksnio pasverta suma
    v1_2 = y1_1*w11_2 + ...
           y2_1*w12_2 + ...
           y3_1*w13_2 + ...
           y4_1*w14_2 + b1_2;

    %% Tinklo isejimas
    y_test(i) = v1_2;

end

%% Tikroji funkcija tik palyginimui grafike
d_test = ((1 + 0.6*sin(2*pi*x_test/0.7)) + ...
          0.3*sin(2*pi*x_test))/2;

%% PAGRINDINIS GRAFIKAS

figure;

plot(x_test,d_test,'--','LineWidth',1.5);
hold on;

plot(x_test,y_test,'LineWidth',1.5);

plot(x,d,'o','MarkerSize',6);

grid on;

xlabel('x');
ylabel('y');

legend('Norimas atsakas', ...
       'Neuroninio tinklo atsakas', ...
       'Mokymo taskai', ...
       'Location','best');

title('Funkcijos aproksimacija daugiasluoksniu perceptronu');

%% Mokymo paklaidos grafikas

figure;

plot(1:length(MSE_history),MSE_history,'LineWidth',1.5);

grid on;

xlabel('Epocha');
ylabel('MSE');

title('Mokymo paklaida');
