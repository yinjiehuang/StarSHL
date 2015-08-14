function bcode = hashing(train,test,learningtype,model,ker_para)
% This function is used to transform the data into binay code (Hash code)
% Input:
%       train           - a structure containing training data
%       test            - a structure containing testing data
%       learningtype    - learning type
%       model           - a structure containing all the parameters
%       ker_para        - Parameters of Multiple Kernels
% Output:
%       bcode           - binay / hash code

% We need to first process the training data
switch learningtype
    case 1 % Supervised Learning
        data = train.Ltr;
    case 2 % Semi-Supervised Learning
        data = [train.Ltr,train.ULtr];
    case 3 % Unsupervised Learning
        data = train.ULtr;
    otherwise
        error(['Error!!!No such learning type is supported!!!! HeHe!!!']);
end
Ntr = size(data,1);
B = size(model.MU,1); % B is the number of bits in binary codes
te = test.te;
Nte = size(te,1);
% Initilize all the kernel matrix
for i = 1:ker_para.num
    if i == 1 % Linear kernel
        eval(['ker.K',num2str(i),' = ker_matrix(''linear'',te'',data'');']);
    elseif i == 2 % Polynomial kernel
        eval(['ker.K',num2str(i),' = ker_matrix(''poly'',te'',data'',''b'',ker_para.polybias,''d'',ker_para.polydegree);'])
    else %Gaussian kernels
        eval(['ker.K',num2str(i),' = ker_matrix(''Gaussian'',te'',data'',''s'',ker_para.Gsigma(',num2str(i-2),'));']);
    end
end

bcode = zeros(Nte,B,'single');
for b = 1:B
    K = zeros(Nte,Ntr,'single');
    for k = 1:ker_para.num
        eval(['K = K+model.Theta(k,b)*ker.K',num2str(k),';']);
    end
    bcode(:,b) = sign(K*model.beta(:,b)+repmat(model.d(b)',Nte,1));
end