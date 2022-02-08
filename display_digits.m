% display digits data set
digitimages = loadMNISTImages('Digits_train-images.idx3-ubyte');
imagelabels = loadMNISTLabels('Digits_train-labels.idx1-ubyte');

digits0to9 = [];
for i = 1:length(imagelabels)
    digits0to9(i,1) = i;
    digits0to9(i,2) = imagelabels(i);
end
digits0to9=sortrows(digits0to9,2);

% fig1: 5x5 grid for digits 0-4
f1 = figure;
for digit0to4 = 1:5
    newdigits = digits0to9(digits0to9(:,2)>=digit0to4,:);
    for j = 1:5
        k = newdigits(j,1);
        subplot(5,5,(j+(5*(digit0to4-1))));
        imshow(digitimages(:,:,k));
    end
end

% fig2: 5x5 grid for digits 5-9
f2 = figure;
for digit5to9 = 1:5
    newdigits = digits0to9(digits0to9(:,2)>=(digit5to9+5),:);
    for j = 1:5
        k = newdigits(j,1);
        subplot(5,5,(j+(5*(digit5to9-1))));
        imshow(digitimages(:,:,k));
    end
end

% display fashion data set
fashionimages = loadMNISTImages('fashion_train-images.idx3-ubyte');
fashionlabels = loadMNISTLabels('fashion_train-labels.idx1-ubyte');

fashionclothing = [];
for i = 1:length(fashionlabels)
    fashionclothing(i,1) = i;
    fashionclothing(i,2) = fashionlabels(i);
end
fashionclothing=sortrows(fashionclothing,2);

% fig3: 5x5 grid for fashion 1-5
f3 = figure;
for item0to4 = 1:5
    newitem = fashionclothing(fashionclothing(:,2)>=item0to4,:);
    for j = 1:5
        k = newitem(j,1);
        subplot(5,5,(j+(5*(item0to4-1))));
        imshow(fashionimages(:,:,k));
    end
end

% fig4: 5x5 grid for fashion 6-10
f4 = figure;
for item5to9 = 1:5
    newitem = fashionclothing(fashionclothing(:,2)>=(item5to9+5),:);
    for j = 1:5
        k = newitem(j,1);
        subplot(5,5,(j+(5*(item5to9-1))));
        imshow(fashionimages(:,:,k));
    end
end