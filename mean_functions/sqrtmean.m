% Given a GP prior belief (specified by the inputs) on sqrt(f), this method 
% provides a GPML-compatible way of computing the induced mean of f(x) (not 
% sqrt(f(x))) for an arbitrary set of locations x. This method also returns 
% the gradient of this mean w.r.t. the GP hyperparameters for the purposes 
% of hyperparameter optimization. Note that (in general) this function should 
% never have to be called directly; it mostly serves as a helper function to 
% sqrtinfExact.m
% 
% Inputs:  cov = covariance function, 
% 		   mn = mean function, 
% 		   hyp = hyperparameter struct of the form ('mean',[mean hyperparameters],'cov',[cov hyperparameters],'lik',[likelihood hyperparameters]),
%		   x = locations where mean function is to be evaluated,
% 		   i = (optional) index of the hyperparameter whose partial derivative is to be evaluated
% Outputs: A = mean of the Chi^2-process evaluated at locations x
% 
% Copyright (c) 2018 Henry Chai.

function A = sqrtmean(cov,mn,hyp,x,i)
	if nargin < 4
		error('Not enough inputs to sqrtmean')
		return;
	end

	n = size(x,1);									% number of input points in x
	D = length(hyp.cov);							% number of covariance hyperparameters
	m = length(hyp.mean); 							% number of mean hyperparameters
	sn2 = exp(2*hyp.lik);							% g-space noise parameter
	mu = feval(mn{:},hyp.mean,x); 					% mean of the g-space GP at x
	sigma = feval(cov{:},hyp.cov,x,'diag')+sn2; 	% covariance of the g-space GP at x

	if nargin == 4
		A = 0.5*(mu.^2+sigma);							% moment-matched mean of a Chi^2 process (http://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature.pdf)
	else
		if i <= m
			A = mu.*feval(mn{:},hyp.mean,x,i);			% derivative of the mean function w.r.t. mean hyperparameters  
		elseif i <= m+D
			A = 0.5*feval(cov{:},hyp.cov,x,'diag',i-m); % derivative of the mean function w.r.t. covariance hyperparameters  
		elseif i == m+D+1
			A = sn2*ones(size(mu));						% derivative of the mean function w.r.t. likelihood hyperparameters  
		else 
			error('Unknown hyperparameter for derivative')
	  	end
	end
end
