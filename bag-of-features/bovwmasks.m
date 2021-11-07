% Code Referenced From Mathwork - 'Image Category Classification using Bag-of-Features 
% Link: https://au.mathworks.com/help/vision/ug/image-category-classification-using-bag-of-features.html
% For all intensive purposes refer to link for more detailed explantations
% of the code.
% Developer: Renz Sinchongco

%Establish Directory & Set the Image Database
setDir  = fullfile('C:\Users\RENZ\Desktop\IPPR\IPPR-A2-Mask-Recognition\');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');

%Check if we set each category
tbl = countEachLabel(imds);

%Used for debugging to check the data storage
%figure
%montage(imds.Files(1:16:end))

%Split the dataset into training and validation. Middle parameter changes
%according to the ratio we filter into training set to validation
[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomize');

%Set the model
bag = bagOfFeatures(trainingSet);

%For analysis purposes, we want to get occurences of vocab which get mapped
%to feature vector.
img = readimage(imds, 1);
featureVector = encode(bag, img);

%Build histogram for visual word occurences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')


categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

%Debug to test model.. Should be near 100% Accuracy with trainingSet
%confMatrix = evaluate(categoryClassifier, trainingSet);

%Establish Confusion Matrix and retrieve average accuracy
confMatrix = evaluate(categoryClassifier, validationSet);
mean(diag(confMatrix))

% To test model against individual images, we preload an image from the
% archive and use predict() to feed into the model
img = imread(fullfile('archive','DatasetSmaller','mask on', '2.png'));
figure
imshow(img)
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the predicted image as a string
categoryClassifier.Labels(labelIdx)