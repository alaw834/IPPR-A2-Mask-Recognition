setDir  = fullfile('C:\Users\RENZ\Desktop\IPPR\IPPR-A2-Mask-Recognition\'); % Main Folder
imgSets = imageSet(setDir,'recursive');

trainingSets = partition(imgSets,2);

bag = bagOfFeatures(trainingSets,'Verbose',false);

img = read(imgSets(1),1);
featureVector = encode(bag,img);