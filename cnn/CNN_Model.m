%Load the dataset into an image datastore and divide 70% of it for training
%and 30% of it for validation
imds = imageDatastore('DatasetSmaller', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

%Load the pre-trained mobilenetv2 network
net = mobilenetv2;

inputSize = net.Layers(1).InputSize;

%Convert the trained network into a layer graph
lgraph = layerGraph(net);

%Find the last learnable layer and the last classification layer that was
%used in the pretrained model
%We want to remove these so we can train the model on a new dataset
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%Comment Below
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%Comment Below
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


%Comment Below
layers = lgraph.Layers;
connections = lgraph.Connections;

lgraph = createLgraphUsingConnections(layers,connections);

%Comment Below
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%Comment Below
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%Comment Below
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Comment Below
net = trainNetwork(augimdsTrain,lgraph,options);

%Comment Below
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%Enables the Model to be used with a webcam for video classification
webcamlist;
cam = webcam;
preview(cam);

inputSize = net.Layers(1).InputSize(1:2)

h = figure;

while ishandle(h)
    im = snapshot(cam);
    image(im)
    im = imresize(im,inputSize);
    [label,score] = classify(net,im);
    title({char(label), num2str(max(score),2)});
    drawnow
end
