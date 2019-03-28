% Given a GP prior belief (specified by the inputs) on sqrt(f), this method 
% provides a GPML-compatible way of computing the induced mean of f(x) (not 
% sqrt(f(x))) for an arbitrary set of locations x. This method also returns 
% the gradient of this mean w.r.t. the GP hyperparameters for the purposes 
% of hyperparameter optimization. Note that (in general) this function should 
% never have to be called directly; it mostly serves as a helper function to 
% sqrtinfExact.m
% 
% Inputs:  cov = covariance function, 
% 		   mean = mean function, 
% 		   hyp = hyperparameter struct of the form ('mean',[mean hyperparameters],'cov',[cov hyperparameters],'lik',[likelihood hyperparameters]),
%		   x = locations where mean function is to be evaluated,
% 		   i = (optional) index of the hyperparameter whose partial derivative is to be evaluated
% Outputs:  K = covariance matrix of the Chi^2 process evaluated at locations x and z
% 
% Copyright (c) 2018 Henry Chai.

function K = sqrtcov(cov,mn,hyp,x,z,i)
	if nargin < 4
		error('Not enough inputs to sqrtcov')
		return;
	end

	if nargin < 5
		z = [];
	end
	xeqz = isempty(z);						% if z is not provided, evaluate the covariance function at x,x
	dg = strcmp(z,'diag');					% if z == 'diag', only return the variances at x
	
	n = size(x,1);							% number of input points in x
	D = length(hyp.cov);					% number of covariance hyperparameters
	m = length(hyp.mean); 					% number of mean hyperparameters
	sn2 = exp(2*hyp.lik);					% g-space noise parameter
	mu = feval(mn{:},hyp.mean,x); 			% mean of the g-space GP at x
	sigma = feval(cov{:},hyp.cov,x); 		% covariance of the g-space GP at x
	sigma = (sigma+sigma')/2+sn2*eye(n);	% make sure the covariance matrix is symmetric and incorporate noise

	if nargin < 6
		if dg
			K = 0.5*diag(sigma).^2+mu.^2.*diag(sigma);											% moment-matched covariance of a Chi^2 process (http://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature.pdf)
		else
			K = 0.5*sigma.^2+(mu*mu').*sigma;													% moment-matched covariance of a Chi^2 process (http://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature.pdf)
		end
	else
		if i <= m
			K = feval(mn{:},hyp.mean,x,i).*(sigma*mu+mu'*sigma);								% derivative of the covariance function w.r.t. mean hyperparameters 
		elseif i <= m+D
			K = sigma.*feval(cov{:},hyp.cov,x,[],i-m)+(mu*mu').*feval(cov{:},hyp.cov,x,[],i-m);	% derivative of the covariance function w.r.t. covariance hyperparameters 
		elseif i == m+D+1'
			K = diag(2*sn2*(diag(sigma)+mu.^2));												% derivative of the covariance function w.r.t. likelihood hyperparameters
		else 
			error('Unknown hyperparameter for derivative')
	  	end
	end
end
