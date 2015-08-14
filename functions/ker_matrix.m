function K = ker_matrix(ker,X1,X2,varargin)
% This function is used to compute the Kernel matrix based on the training
% samples:
% Input:
%       - ker, string, name of the Kernels, this function supports three
%       types of Kernels: 'linear', 'poly', 'Gaussian'
%       - X1, data matrix 1, features times # of training data
%       - X2, data matrix 2, features times # of training data
%       Parameters:
%       - b, bias of the polynomial Kernel, default = 0
%       - d, degree of the polynomial Kernel, default = 3
%       - s, sigma of the Gaussian Kernel, default = 1
% Output:
%       - K, Kernel Matrix


% Set default parameters
pars.b = 1;
pars.d = 2;
pars.s = 1;

% Pass the parameters in the function if they exist
for i =1:2:length(varargin)
    if i < length(varargin)
        eval(['pars.',varargin{i},'=',num2str(varargin{i+1}),';']);
    end
end

% Now start computing kernel matrix
switch ker
    case 'linear'
        K = X1'*X2;
        % Let's normalize the kernel
        temp = sqrt(diag(X1'*X1));
        norm1 = repmat(temp,1,size(X2,2));
        temp = sqrt(diag(X2'*X2));
        norm2 = repmat(temp',size(X1,2),1);
        clear temp;
        K = K./(norm1.*norm2);
        clear norm1 norm2;
    case 'poly'
        K = (X1'*X2+pars.b).^pars.d;
        % Let's normalize the kernel
        tempK = (X1'*X1+pars.b).^pars.d;
        temp = sqrt(diag(tempK));
        norm1 = repmat(temp,1,size(X2,2));
        tempK = (X2'*X2+pars.b).^pars.d;
        temp = sqrt(diag(tempK));
        norm2 = repmat(temp',size(X1,2),1);
        clear temp tempK;
        K = K./(norm1.*norm2);
        clear norm1 norm2;
    case 'Gaussian'
        E = bsxfun(@plus,sum(X1.*X1,1)',(-2)*X1'*X2);
        E = bsxfun(@plus,sum(X2.*X2,1),E);
        K = exp(-E/(2*pars.s^2));
        clear E;
    otherwise
        error(['Sorry, we do not support this kernel: ',ker]);
end