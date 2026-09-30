function P = extract_symbol_features(imageName, numberOfRows, showSteps)
% P = extract_symbol_features(imageName, numberOfRows, showSteps)
%
% Is ranka rasytu simboliu paveikslelio isskiria po 35 pozymius.
%
% Kiekvienas simbolis:
% 1) segmentuojamas,
% 2) apkerpamas,
% 3) pakeiciamas i 70x50,
% 4) padalinamas i 7x5 blokus po 10x10,
% 5) kiekviename bloke skaiciuojama rasalo dalis.
%
% Rezultatas P:
%   35 x simboliu_skaicius

if nargin < 3
    showSteps = true;
end

I = imread(imageName);

if size(I,3) == 3
    G = rgb2gray(I);
else
    G = I;
end

% Binarizacija. Baltas fonas = 1, tamsus rasalas = 0.
level = graythresh(G);
BW = imbinarize(G,level);

% Segmentacijai rasalas turi buti foreground = 1.
ink = ~BW;

% Nedidelis sujungimas, kad vieno simbolio dalys susijungtu.
se = strel('square',3);
ink = imclose(ink,se);
ink = imdilate(ink,strel('square',2));
ink = imfill(ink,'holes');
ink = bwareaopen(ink,10);

CC = bwconncomp(ink);
stats = regionprops(CC,'BoundingBox','Centroid','Area');

if isempty(stats)
    error('Paveikslelyje nerasta simboliu.');
end

% Atmetame labai mazus triuksmo objektus pagal plota.
areas = [stats.Area];
medianArea = median(areas);
keep = areas > max(10,0.10*medianArea);

stats = stats(keep);

N = length(stats);

if mod(N,numberOfRows) ~= 0
    error(['Rasta %d objektu, bet skaicius nesidalina is eiluciu skaiciaus %d.\n' ...
           'Patikrinkite tarpus tarp simboliu arba numberOfRows.'], ...
           N, numberOfRows);
end

symbolsPerRow = N/numberOfRows;

centers = reshape([stats.Centroid],2,[])';

% Pirmiausia rusiuojame pagal Y, kad gautume eilutes.
[~,orderY] = sort(centers(:,2));
orderY = orderY(:);

order = zeros(N,1);

% Kiekvienoje eiluteje rusiuojame is kaires i desine pagal X.
for row = 1:numberOfRows

    ids = orderY((row-1)*symbolsPerRow+1:row*symbolsPerRow);

    [~,localOrder] = sort(centers(ids,1));

    order((row-1)*symbolsPerRow+1:row*symbolsPerRow) = ...
        ids(localOrder);

end

P = zeros(35,N);

if showSteps
    figure;
end

for k = 1:N

    box = stats(order(k)).BoundingBox;

    symbolInk = imcrop(ink,box);

    % Apkerpame tuscias krastines.
    rowsWithInk = any(symbolInk,2);
    colsWithInk = any(symbolInk,1);

    symbolInk = symbolInk(rowsWithInk,colsWithInk);

    % Visi simboliai vienodo dydzio.
    symbolInk = imresize(symbolInk,[70 50]);

    if showSteps
        subplot(numberOfRows,symbolsPerRow,k);
        imshow(symbolInk);
    end

    featureNumber = 1;

    % 7 x 5 = 35 pozymiai
    for m = 1:7
        for n = 1:5

            block = symbolInk((m-1)*10+1:m*10, ...
                              (n-1)*10+1:n*10);

            % Vidutine rasalo dalis bloke, intervalas 0...1
            P(featureNumber,k) = mean(block(:));

            featureNumber = featureNumber + 1;

        end
    end

end

if showSteps
    sgtitle(['Segmentuoti simboliai: ', imageName]);
end

end
