function hmodel = learningHash(train, learningtype, MUin, Thetain, ker_para)
% This function learns the Hash function based on our algorithm:
% Input:
%       train     - a structure contains the training data
%       learningtype - an integer (1~3) indicates the learning type
%                       1 - Supervised Learning
%                       2 - Semi-Supervised Learning
%                       3 - Unsupervised Learning
%       MUin         - Codewords / prototypes (pre_trained)
%       Thetain      - Weights of Multiple Kernel Learning (pre_trained)
%       ker_para     - Parameters of Multiple Kernels
% Output:
%       hmodel        - a structure containing all the trained parameters:
%                      beta, d, MU, Theta

threshold = 0.05;
addpath([pwd, '\libsvm']);
opt = '-c 1000 -t 4 -q';
p = 2;

% We need to first process the training data
switch learningtype
    case 1 % Supervised Learning
        data = train.Ltr;
    case 2 % Semi-Supervised Learning
        data = [train.Ltr, train.ULtr];
    case 3 % Unsupervised Learning
        data = train.ULtr;
    otherwise
        error(['Error!!!No such learning type is supported!!!! HeHe!!!']);
end
N = size(data, 1);
[B, C] = size(MUin); % B is the number of bits in binary codes

% Initilize all the kernel matrix
for i = 1:ker_para.num
    if i == 1 % Linear kernel
        eval(['ker.K', num2str(i), ' = ker_matrix(''linear'',data'',data'');']);
    elseif i == 2 % Polynomial kernel
        eval(['ker.K', num2str(i), ' = ker_matrix(''poly'',data'',data'',''b'',ker_para.polybias,''d'',ker_para.polydegree);'])
    else %Gaussian kernels
        eval(['ker.K', num2str(i), ' = ker_matrix(''Gaussian'',data'',data'',''s'',ker_para.Gsigma(', num2str(i-2), '));']);
    end
end

%Supervised Learning gamma vector could be pre caculated
if learningtype == 1 
    gamma = comGammaL(train);
    index0 = single(find(gamma == 0));
    indexN0 = single(find(gamma ~= 0));
    
end
clear train
ytilta = zeros(N, B, 'single');
beta = zeros(N, B, 'single');
din = zeros(B, 1, 'single');
w_square = zeros(ker_para.num, B, 'single');
Thetain_U = zeros(ker_para.num, B, 'single');
while (1)
    % Compute the gamma vector
    switch learningtype
        case 1
            ;
        case 2 %Semi-supvised Learning needs to be 
            ;
        case 3
            ;
        otherwise
            error(['Error!!!No such learning type is supported!!!! HeHe!!!']);
    end
    for b = 1:B %For each bit, we will do hashing
        % Let's add some screen information 
        if b == 1
            fprintf('Bit: %d', b);
        elseif mod(b, 10) == 0
            fprintf(', %d\n', b);
        else
            fprintf(', %d', b);
        end        
        label = kron(MUin(b, :)', ones(N, 1, 'single'));
        labelU = label(indexN0);
        clear label;
        %Compute the Kernerl matrix
        K = zeros(N, N, 'single');
        for k = 1:ker_para.num
            eval(['K = K+Thetain(k,b)*ker.K', num2str(k), ';']);
        end
        
        % In order to reduce memory load, change it to loops
        PerN = N / C;
        temp1 = zeros(N, N);
        index = 1;
        for c = 1:C
            tempindex = find(index0 > (c - 1) * N & index0 <= c * N);
            eval(['N0struct.n0_', num2str(c), ' = tempindex;']);
            tempK = K;
            tempK(:, index0(tempindex) - (c - 1) * N) = [];
            indexup = index + size(tempK, 2) - 1;
            temp1(:, index:indexup) = tempK;
            index = indexup + 1;
        end
        clear tempK;
        BigKU = zeros(N, N);
        index = 1;
        for c = 1:C
            eval(['tempindex = N0struct.n0_',num2str(c),';']);
            tempK = temp1;
            tempK(index0(tempindex)-(c-1)*N,:) = [];
            indexup = index+size(tempK,1)-1;
            BigKU(index:indexup,:) = tempK;
            index = indexup+1;
        end
        clear tempindex temp1 tempK N0struct;
        
        % Step 1, solve SVM
        Num_libsvm = size(BigKU, 1);
        PreKer_tr = [(1:Num_libsvm)', BigKU];
        % I have to use doulbe presicion which is requred by LIBSVM
        model = svmtrain(double(labelU), double(PreKer_tr), opt);
        clear PreKer_tr labelU;
        alpha_s = zeros(Num_libsvm, 1, 'single');
        alpha_s(model.SVs) = model.sv_coef;
        alpha = zeros(N * C, 1, 'single');
        alpha(indexN0) = alpha_s;
        clear alpha_s;
        beta(:, b) = kron(ones(1, C, 'single'), eye(N, 'single')) * alpha;
        clear alpha;
        din(b) = - model.rho;
        clear model;
        ytilta(:, b) = K * beta(:, b) + din(b) * ones(N, 1, 'single');
        clear BigKU K;
        % Step 2, Update MKL parameters
        % First compute the weight square
        for k = 1:ker_para.num
            eval(['w_square(k,b) = Thetain(k,b)^2*beta(:,b)''*ker.K', num2str(k), '*beta(:,b);']);
        end
        Thetasum = sum(w_square(:, b) .^ (p / (1 + p))) .^ (1 / p);
        Thetain_U(:, b) = w_square(:, b) .^ (1 / (1 + p)) / Thetasum;
        % Step 3, Update codewords
        gamma_m = reshape(gamma, N, C);
        for i = 1:C
            term1 = gamma_m(:, i)' * double((sign(ytilta(:, b)) ~= ones(N, 1))); % mu is +1
            term2 = gamma_m(:, i)' * double((sign(ytilta(:, b)) ~= - ones(N, 1))); % mu is -1
            if term1 <= term2
                MUin(b, i) = 1;
            else
                MUin(b, i) = -1;
            end
        end
        temp = MUin(b, :);
        if max(temp) == min(temp)
            MUin(b, randi(C, 1, 'single')) = - MUin(b, randi(C, 1, 'single'));
        end
    end
    diff = max(max(abs(Thetain_U - Thetain)));
    fprintf('\n Maximum difference - MKL Theta: %f. \n', diff);
    Thetain = Thetain_U;
    if diff <= threshold
        break;
    end
end
hmodel.Tr_bcode = sign(ytilta);
hmodel.beta = beta;
hmodel.d = din;
hmodel.MU = MUin;
hmodel.Theta = Thetain;

function gamma = comGammaL(train)
% This function is used to calculate the gamma vector for labelled data
% Input:
%       train - training data including data and labels
% Output:
%       gamma - gamma vector (size NC X 1)

% Extract C - number of classes
C = length(unique(train.Ltr_L));
Num = size(train.Ltr, 1);

left = repmat(train.Ltr_L, C, 1);
temp = repmat((0:(C - 1)), Num, 1);
right = temp(:);
gamma = single(left == right);