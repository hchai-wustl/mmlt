% Given a GP prior belief (specified by the inputs) on log(f), this method 
% provides a GPML-compatible way of computing the induced covariance between 
% f(x) and f(z) (not log(f(x)) and log(f(z))) for arbitrary sets of locations 
% x and z. This method also returns the gradient of this covariance w.r.t. the 
% GP hyperparameters for the purposes of hyperparameter optimization. Note that 
% (in general) this function should never have to be called directly; it mostly
% serves as a helper function to loginfExact.m
% 
% Inputs:  cov = covariance function, 
% 	 	   mean = mean function, 
% 		   hyp = hyperparameter struct of the form ('mean',[mean hyperparameters],'cov',[cov hyperparameters],'lik',[likelihood hyperparameters]),
% 		   x,z (optional) = pair of points where the covariance function is to be evaluated
% 		   i (optional) = index of the hyperparameter whose partial derivative is to be evaluated
% Outputs: K = covariance matrix of the log-normal process evaluated at locations x and z
% 
% Copyright (c) 2018 Henry Chai.

function K = logcov(cov,mn,hyp,x,z,i)
	if nargin < 4
		error('Not enough inputs to logcov')
		return; 
	end       
	
	if nargin < 5
		z = []; 
	end                                  
	xeqz = isempty(z);						% if z is not provided, evaluate the covariance function at x,x
	dg = strcmp(z,'diag');					% if z == 'diag', only return the variances at x
	
	n = size(x,1);							% number of input points in x
	D = length(hyp.cov);					% number of covariance hyperparameters
	m = length(hyp.mean);					% number of mean hyperparameters
	sn2 = exp(2*hyp.lik);					% g-space noise parameter
	mu = feval(mn{:},hyp.mean,x);			% mean of the g-space GP at x
	sigma = feval(cov{:},hyp.cov,x);		% covariance of the g-space GP at x
	sigma = (sigma+sigma')/2+sn2*eye(n);	% make sure the covariance matrix is symmetric and incorporate noise
	
	if nargin < 6                                                    
		if dg 
			K = exp(2*mu+2*diag(sigma))-exp(2*mu+diag(sigma));	% moment-matched covariance of a log-normal process (https://en.wikipedia.org/wiki/Log-normal_distribution)
		else
			pwr = bsxfun(@plus,sigma,mu);						% use bsxfun to efficiently compute matrix operations on differently sized matrices
			pwr = bsxfun(@plus,pwr,mu');
			pwr = bsxfun(@plus,pwr,0.5*diag(sigma));
			pwr = bsxfun(@plus,pwr,0.5*diag(sigma)');
			
			K = exp(pwr)-exp(pwr-sigma);						% moment-matched covariance of a log-normal process (https://en.wikipedia.org/wiki/Log-normal_distribution)
		end 
	else     
		pwr = bsxfun(@plus,sigma,mu);
		pwr = bsxfun(@plus,pwr,mu');
		pwr = bsxfun(@plus,pwr,0.5*diag(sigma));
		pwr = bsxfun(@plus,pwr,0.5*diag(sigma)');
		
		if i <= m
			K = (exp(pwr)-exp(pwr-sigma)).*(feval(mn{:},hyp.mean,x,i)+feval(mn{:},hyp.mean,x,i)');		% derivative of the covariance function w.r.t. mean hyperparameters 
		elseif i <= m+D
			dK_diag = 0.5*feval(cov{:},hyp.cov,x,'diag',i-m)+0.5*feval(cov{:},hyp.cov,x,'diag',i-m)';
			K = (exp(pwr).*(dK_diag+feval(cov{:},hyp.cov,x,[],i-m)))-(exp(pwr-sigma).*dK_diag);    		% derivative of the covariance function w.r.t. covariance hyperparameters 
		elseif i == m+D+1
			K = diag(2*sn2*(2*exp(diag(pwr))-exp(diag(pwr-sigma))));    								% derivative of the covariance function w.r.t. likelihood hyperparameters
		else 
			error('Unknown hyperparameter for derivative')
	  	end
	end
end