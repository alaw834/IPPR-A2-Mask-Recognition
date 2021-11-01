setDir  = fullfile('C:\Users\RENZ\Desktop\IPPR\IPPR-A2-Mask-Recognition\');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');

extractor = @exampleBagOfFeaturesExtractor;
bag = bagOfFeatures(imds,'CustomExtractor',extractor)