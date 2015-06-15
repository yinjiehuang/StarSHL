%% This is the implementation of Our Hashing
clear;
clc;

tic;
%% Load the data and preprocessing the dataset
fprintf('Preprocessing the data and initialize the parameters......\n');

% In the dataset, the data structure is like (# of data X features):
%       Ltr     - labelled training data
%       Ltr_L   - labels for labelled training data
%       ULtr    - unlablelled training data
%       te      - testing data
%       te_L    - labels for testing data

% Please incicate what kind of learning:
%       1   - Supervised Learning (Ltr, Ltr_L, te, te_L)
%       2   - Semi-supervised Learning (Ltr, Ltr_L, ULtr, te, te_L)
%       3   - Unsupervised Learning (ULtr, te, te_L)
Learning_Type = 1; 

currentpath = cd('..');
parentpath = pwd();
cd(currentpath);

addpath([parentpath,'/Datasets/Mnist_3000_6000']);
addpath([parentpath,'/Datasets/']);

% Initialize the result matrix
result_Our = zeros(5,9,'single');

% Indicate how many runs/epoches you need
for epoch = 1:1
    fprintf('Now we are running %d epoch......\n\n',epoch);
    
%     eval(['load Mnist_', num2str(epoch), ';']);
    load small;
%       load small_m
%     train.Ltr = single(train.Ltr);
%     train.Ltr_L = single(train.Ltr_L);
%     test.te = single(test.te);
%     test.te_L = single(test.te_L);
    
    [Num,Dim] = size(test.te);

    for B = 45:5:45
    % B = 10;     % Number of bits

    %% Codewords / Prototypes need to be initialized 
    switch Learning_Type
        case 1 || 2 % Supervised Learning / Semi-Supervised Learning
            C = length(unique(train.Ltr_L)); % Num of classes or clusters
        case 3 % Unsupervised Learning
            % Number of codewords needs to be pre-defined
            C = 4; 
        otherwise
            error(['Error!!!No such learning type is supported!!!! HeHe!!!']);
    end
    MU = sign(rand(B,C,'single')-0.5); % Randomly intialized the codewords
    % if there is some zero involved in the codewords, replace them
    [in0x,in0y] = find(MU == 0);
    if ~isempty(find(MU == 0))
        MU(in0x,in0y) = -1;
    end
    % Every bit, at least one element is different
    for b = 1:B
        temp = MU(b,:);
        if max(temp) == min(temp)
            MU(b,randi(C,1,'single')) = -MU(b,randi(C,1,'single'));
        end
    end
    clear in0x in0y temp;

    %% Weights for Multiple Kernel Learning
    % Number of Kernels involved (1 linear, 1 polynomial, 9 Gaussians)
    ker_para.num = 11;
    ker_para.polybias = 1; % Bias of polynomial kernels
    ker_para.polydegree = 2; % Degree of polynomial kernels
    % Sigma of Gaussian Kernels
    Sigamas = [-7,-5,-3,-1,0,1,3,5,7];
    ker_para.Gsigma = 2.^Sigamas;
    % Initilize the theta
    Theta = rand(ker_para.num,B,'single');
    Theta = Theta./repmat(sum(Theta,1),ker_para.num,1);
    % Predefine the Kernel 

    %% Train the model
    fprintf('Begin training the model: \n');
    model = learningHash(train,Learning_Type,MU,Theta,ker_para);

    %% Retrivel
    fprintf('\n\nTesting Phase!!!!!\n');
    hashcode = hashing(train,test,Learning_Type,model,ker_para);
    % The proportion of true neighbors in top-k retrieval 
    top_k = 10;
    B1 = compactbit((model.Tr_bcode+1)/2);
    B2 = compactbit((hashcode+1)/2);
    Dhamm = hammingDist(B2,B1);
%     eval(['save Our_HM',num2str(epoch),'_',num2str(B),' Dhamm']);

    [Num_te,Ndimension] = size(test.te);
    Num_tr_nei_B = zeros(Num_te,1,'single');
    for j = 1:Num_te
        [sorted,index] = sort(Dhamm(j,:));
        retri_L = train.Ltr_L(index(1:top_k));
        Num_tr_nei_B(j) = sum(retri_L == repmat(test.te_L(j),top_k,1));
    end
    result_Our(epoch,B/5) = sum(Num_tr_nei_B)/(top_k*Num_te);
    fprintf('\n*********Result: Epoch %d, %d Bits, Retrieval Accuracy: %f.*********\n\n',epoch,B,result_Our(epoch,B/5));
    end
end
toc;