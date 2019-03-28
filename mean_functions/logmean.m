% Given a GP prior belief (specified by the inputs) on log(f), this method 
% provides a GPML-compatible way of computing the induced mean of f(x) (not 
% log(f(x))) for an arbitrary set of locations x. This method also returns 
% the gradient of this mean w.r.t. the GP hyperparameters for the purposes 
% of hyperparameter optimization. Note that (in general) this function should 
% never have to be called directly; it mostly serves as a helper function to 
% loginfExact.m
% 
% Inputs:  cov = covariance function, 
% 		   mn = mean function, 
% 		   hyp = hyperparameter struct of the form ('mean',[mean hyperparameters],'cov',[cov hyperparameters],'lik',[likelihood hyperparameters]),
%		   x = locations where mean function is to be evaluated,
% 		   i = (optional) index of the hyperparameter whose partial derivative is to be evaluated
% Outputs: A = mean of the log-normal process evaluated at locations x
% 
% Copyright (c) 2018 Henry Chai.

function A = logmean(cov,mn,hyp,x,i)
	if nargin < 4 
		error('Not enough inputs to logmean')
		return; 
	end           
	
	n = size(x,1);									% number of input points in x
	D = length(hyp.cov);							% number of covariance hyperparameters
	m = length(hyp.mean);							% number of mean hyperparameters
	sn2 = exp(2*hyp.lik);							% g-space noise parameter
	mu = feval(mn{:},hyp.mean,x);					% mean of the g-space GP at x
	sigma = feval(cov{:},hyp.cov,x,'diag')+sn2;		% covariance of the g-space GP at x
	
	if nargin == 4
		A = exp(mu+0.5*sigma);												 % moment-matched mean of a log-normal process (https://en.wikipedia.org/wiki/Log-normal_distribution)
	else
		if i <= m
			A = exp(mu+0.5*sigma).*feval(mn{:},hyp.mean,x,i);  				 % derivative of the mean function w.r.t. mean hyperparameters  
		elseif i <= m+D
			A = exp(mu+0.5*sigma).*(0.5*feval(cov{:},hyp.cov,x,'diag',i-m)); % derivative of the mean function w.r.t. covariance hyperparameters  
		elseif i == m+D+1
			A = exp(mu+0.5*sigma).*sn2;										 % derivative of the mean function w.r.t. likelihood hyperparameters  
		else 
			error('Unknown hyperparameter for derivative')
	  	end
	end
end