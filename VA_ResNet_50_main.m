
% change to your own data directory .mat format with matrix size 1280 x 100 x 3 
training_folder = 'directory of training data';
testing_folder = 'directory of testing data';

trainingfiles = imageDatastore(training_folder,"ReadFcn",@matRead,"FileExtensions",".mat",'IncludeSubfolders',1,'FileExtensions','.mat','LabelSource', 'foldernames');
testfiles = imageDatastore(testing_folder,"ReadFcn",@matRead,"FileExtensions",".mat",'IncludeSubfolders',1,'FileExtensions','.mat','LabelSource', 'foldernames');
%%
% Load the pre-trained ResNet-50 from Matlab.
net = resnet50;
% Get the layer graph.
lgraph = layerGraph(net);
% Create a new image input layer of the desired size.
inputSize = [100 1280 3];
newInputLayer = imageInputLayer(inputSize,'Name','new_input','Normalization','zerocenter');
lgraph = replaceLayer(lgraph,'input_1', newInputLayer);
% Remove the last three layers
lgraph = removeLayers(lgraph, {'fc1000_softmax','ClassificationLayer_fc1000'});
% Add new layers
newLayers = [
    dropoutLayer(0.6,"Name","dropout_1")
    reluLayer("Name","relu1")
     fullyConnectedLayer(400,"Name","fc_1")
    % reluLayer("Name","relu_6")
    dropoutLayer(0.6,"Name","dropout_2")
    reluLayer("Name","relu2")
    % softmaxLayer('Name','softmax1')
    fullyConnectedLayer(2,'Name','fc') % 
    % dropoutLayer(0.55, 'Name', 'new_dropout1')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];
% Connect the new layer to the layer before the removed layer
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'fc1000', 'dropout_1');
%% train model
ilr = 0.00001;
mxEpochs = 20;
mbSize = 32;

opts = trainingOptions('adam', 'InitialLearnRate', ilr, ...
    'MaxEpochs',mxEpochs , 'MiniBatchSize',mbSize, ...
    'Plots', 'training-progress','Shuffle','once',...
    'L2Regularization',0.01, ... % Add L2 regularization
    'ValidationData',{testfiles,testfiles.Labels},'ExecutionEnvironment','multi-gpu');

[myNet, info] = trainNetwork(trainingfiles, lgraph, opts); %myNe

save('trainedNetwork.mat', 'myNet');
%% prediction

testfiles.ReadFcn = @matRead;
[predictedLabels,scores] = classify(myNet, testfiles);
accuracy = mean(predictedLabels == testfiles.Labels)
[confmat, order] = confusionmat(testfiles.Labels,predictedLabels);
figure
heatmap(order,order,confmat);
figure
confusionchart(testfiles.Labels,predictedLabels,'Normalization','row-normalized','RowSummary','row-normalized')
%%
%%
function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
% data=reshape(data,100,1280,1);
% data=repmat(data,[1,1,3]);
end


function data = matRead2(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
% data=reshape(data,100,1280,1);
data = imresize(data, [227 227]);
data=repmat(data,[1,1,3]);
end

function lgraph = copyWeights(net, lgraph)
    for i = 1:numel(net.Layers)
        if isprop(net.Layers(i), 'Weights')
            lgraph = replaceLayer(lgraph, net.Layers(i).Name, net.Layers(i));
        end
    end
end