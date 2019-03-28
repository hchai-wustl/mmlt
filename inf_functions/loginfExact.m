% Given a GP prior belief (specified by the inputs) on log(f), a set of 
% observations (x,log(f(x))) and a setting of the GP hyperparameters, 
% this method provides a GPML-compatible way of computing the (negative 
% log) likelihood of the observations (x,f(x)) (not (x,log(f(x)))) under 
% the input hyperparameter setting. This method also returns the gradient 
% of the (negative log) likelihood w.r.t. the GP hyperparameters for the 
% purposes of hyperparameter optimization. 
% 
% This function is compatible with the minimize.m function provided by 
% the GPML toolbox. For an example of the correct usage of this function, 
% see example.m 
% 
% Note that the current implementation of this function is only compatible
% with additive Gaussian noise on the log(f) observations.
% 
% Inputs:  cov = covariance function, 
% 		   mn = mean function, 
% 		   hyp = hyperparameter struct of the form ('mean',[mean hyperparameters],'cov',[cov hyperparameters],'lik',[likelihood hyperparameters]),
%		   x = locations where mean function is to be evaluated,
% 		   i = (optional) index of the hyperparameter whose partial derivative is to be evaluated
% Outputs: post = struct containing objects useful for computing posterior beliefs
% 		   nlZ = negative log likelihood of the observations (x,f(x))
% 		   dnlZ = derivative of the negative log likelihood w.r.t. the GP hyperparameters; has the same form as hyp
%
% Copyright (c) 2018 Henry Chai.

function [post,nlZ,dnlZ] = loginfExact(hyp,mn,cov,lik,x,y)
	if iscell(lik)
		likstr = lik{1};
	else
		likstr = lik;
	end

	if ~ischar(likstr)
		likstr = func2str(likstr);
	end

	if ~strcmp(likstr,'likGauss')
		error('Exact inference only possible with Gaussian likelihood');
	end

	n = size(x,1);							% number of input points in x
	D = length(hyp.cov);					% number of covariance hyperparameters
	m = length(hyp.mean);					% number of mean hyperparameters
	mu = logmean(cov,mn,hyp,x);       		% mean of the f-space GP at x     
	K = logcov(cov,mn,hyp,x);            	% covariance of the f-space GP at x        
	K = (K+K')/2;      						% ensure the covariance matrix is symmetric

	try
		L = chol(K);						% try to compute the Cholesky decomposition of the covariance matrix
	catch									% this sometimes fails due to numerical instability
		[V,W] = eig(K);						% in that case, scale the smallest eigenvalue to be non-negative

		warning('Cov matrix is not PSD, increasing min eigenvalue of %f to 0.000001',min(diag(W)));
		new_diag = diag(W);
		new_diag(new_diag < 1e-6) = 1e-6;
		W = diag(new_diag);
		K = V*W*V';
		K = (K+K')/2;    					% ensure the covariance matrix is symmetric

		L = chol(K);
	end
	pL = -solve_chol(L,eye(n));				% values useful for computing the posterior belief given the observations (x,y)
	alpha = solve_chol(L,exp(y)-mu);		% see GPML chapter 2.2 (http://gaussianprocess.org/gpml/chapters/RW2.pdf)
	
	post.alpha = alpha;						% store these values in the returned object post
	post.sW = ones(n,1);
	post.L = pL;

	if nargout > 1
		nlZ = (exp(y)-mu)'*alpha/2+sum(log(diag(L)))+n*log(2*pi)/2; 					% compute the negative log likelihood of the observations (x,y)
		if nargout > 2																	
			dnlZ = hyp;
			for i = 1:(m+D+1)
				dmu = logmean(cov,mn,hyp,x,i);											% derivative of the f-space mean w.r.t. the i^(th) hyperparameter
				dK = logcov(cov,mn,hyp,x,[],i);											% derivative of the f-space covariance w.r.t. the i^(th) hyperparameter
				Z = -0.5*alpha'*dK*alpha-dmu'*alpha+0.5*sum(diag(solve_chol(L,dK)));	% derivative of the negative log likelihood w.r.t. the i^(th) hyperparameter
				if i <= m																% fill in the appropriate entry of the returned object dlnZ
					dnlZ.mean(i) = Z;	
				elseif i <= m+D
					dnlZ.cov(i-m) = Z;
				elseif i == m+D+1
					dnlZ.lik = Z;
				end
			end
		end
	end
end
