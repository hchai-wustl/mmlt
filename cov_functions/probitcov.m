% Given a GP prior belief (specified by the inputs) on probit(f), this method 
% provides a GPML-compatible way of computing the induced covariance between 
% f(x) and f(z) (not probit(f(x)) and probit(f(z))) for arbitrary sets of 
% locations x and z. This method also returns the gradient of this covariance 
% w.r.t. the GP hyperparameters for the purposes of hyperparameter optimization. 
% Note that (in general) this function should never have to be called directly; 
% it mostly serves as a helper function to probitinfExact.m or probitinfLOO.m
%
% The gradients of the covariance matrix w.r.t. the GP hyperparameters are 
% related to the moments of a truncated, bivariate Gaussian, which are computed
% here using Rosenbaum (1961) (https://www.jstor.org/stable/2984029)
% 
% The current implementation is unfortunately very slow and should only be used 
% for a small number of locations; this is because MATLAB does not have a 
% vectorized implementation of bvncdf (the CDF of a bivariate Gaussian) and so 
% each compenent of the covariance matrix must be computed sequentially. 
% Included at the bottom of this file is also a (slightly) optimized version of 
% MATLAB's internal code for bvncdf.
%
% Inputs:  cov = covariance function, 
% 	 	   mean = mean function, 
% 		   hyp = hyperparameter struct of the form ('mean',[mean hyperparameters],'cov',[cov hyperparameters],'lik',[likelihood hyperparameters]),
% 		   x,z (optional) = pair of points where the covariance function is to be evaluated
% 		   i (optional) = index of the hyperparameter whose partial derivative is to be evaluated
% Outputs: K = covariance matrix of the probit-normal process evaluated at locations x and z
% 
% Copyright (c) 2018 Henry Chai.

function K = probitcov(cov,mn,hyp,x,z,i)
	if nargin < 4
		error('Not enough inputs to probitcov')
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

	M = probitmean(cov,mn,hyp,x);								% Compute the first moments (i.e. the mean) at x
	if nargin < 6
		if dg
			K = NaN(n,1);										
			for j = 1:n
				% K is first populated with the second raw moment
				K(j) = bvncdf([mu(j)/sqrt(sigma(j,j)+1) mu(j)/sqrt(sigma(j,j)+1)],sigma(j,j)/(sigma(j,j)+1));									
			end
			K = K - M.^2;										% Subtracting the first moment squared gives the variance
		elseif xeqz
			K = NaN(n);
			for j = 1:n
				% K is first populated with the second raw moment
				% Note that the diagonal elements are computed slightly differently than the off-diagonal elements
				K(j,j) = bvncdf([mu(j)/sqrt(sigma(j,j)+1) mu(j)/sqrt(sigma(j,j)+1)],sigma(j,j)/(sigma(j,j)+1));									

				for k = 1:(j-1)
					% K is first populated with the second raw moment; these moments are symmetric
					K(j,k) = bvncdf([mu(j)/sqrt(sigma(j,j)+1) mu(k)/sqrt(sigma(k,k)+1)],sigma(j,k)/sqrt((sigma(j,j)+1)*(sigma(k,k)+1)));		
					K(k,j) = K(j,k);																											
				end
			end
			K = K - M*M';										% Subtracting the pairwise product of the first moments gives the covariance
		else
			nz = size(z,1);										% number of input points in z
			muz = feval(mn{:},hyp.mean,z);						% mean of the g-space GP at z
			sigmaz = feval(cov{:},hyp.cov,z,'diag');			% covariance of the g-space GP at z
			sigmaz = (sigmaz+sigmaz')/2+sn2*eye(nz);			% make sure the covariance matrix is symmetric and incorporate noise
			sigmaxz = feval(cov{:},hyp.cov,x,z);				% covariance of the g-space GP between points in x and points in z
			Mz = probitmean(cov,mn,hyp,z);						% Compute the first moments (i.e. the mean) at z

			K = NaN(n,nz);
			for j = 1:n
				for k = 1:nz
					% K is first populated with the second raw moment
					K(j,k) = bvncdf([mu(j)/sqrt(sigma(j,j)+1) muz(k)/sqrt(sigmaz(k,k)+1)],sigmaxz(j,k)/sqrt((sigma(j,j)+1)*(sigmaz(k,k)+1)));	
				end
			end
			K = K - M*Mz';										% Subtracting the pairwise product of the first moments gives the covariance
		end
	else
		if i <= m												% Code for computing the derivative of the covariance function w.r.t. mean hyperparameters 
			dmu = feval(mn{:},hyp.mean,x,i);					% Compute the derivative of the g-space mean w.r.t. mean hyperparameters 
			dM = probitmean(cov,mn,hyp,x,i);					% Compute the derivative of the f-space mean w.r.t. mean hyperparameters 
			
			K = NaN(n);
			for j = 1:n
				for k = 1:j
					sig = [sigma(j,j)+1,sigma(j,k);sigma(k,j),sigma(k,k)+1];	% Compute some relevant quantities; for complete details see Rosenbaum (1961)
					rho = sigma(j,k)/sqrt(sig(1,1)*sig(2,2));					
					rho_adj = sqrt(1-rho^2);
					mu_adj = [mu(j) mu(k)]./sqrt([sig(1,1) sig(2,2)]);

					% K is first populated with the derivative of the second raw moment w.r.t. mean hyperparameters; these derivatives are symmetric
					K(j,k) = dmu(k)*normcdf(mu_adj(1),mu(k)*rho/sqrt(sig(2,2)),rho_adj)*normpdf(mu(k),0,sqrt(sig(2,2)))+dmu(j)*normcdf(mu_adj(2),mu(j)*rho/sqrt(sig(1,1)),rho_adj)*normpdf(mu(j),0,sqrt(sig(1,1)));	
					K(k,j) = K(j,k);							
				end
			end
			K = K-dM*M'-M*dM';									% Subtract the derivative of pairwise product of the first moments (computed using the product rule)
		elseif i <= m+D											% Code for computing the derivative of the covariance function w.r.t. covariance hyperparameters 
			dsigma = feval(cov{:},hyp.cov,x,[],i-m);			% Compute the derivative of the g-space covariance w.r.t. covariance hyperparameters 
			dM = probitmean(cov,mn,hyp,x,i);					% Compute the derivative of the f-space mean w.r.t. covariance hyperparameters 

			K = NaN(n);
			moments = zeros(2);
			for j = 1:n
				for k = 1:j
					sig = [sigma(j,j)+1,sigma(j,k);sigma(k,j),sigma(k,k)+1];	% Compute some relevant quantities; for complete details see Rosenbaum (1961)
					rho = sigma(j,k)/sqrt(sig(1,1)*sig(2,2));
					rho_adj = sqrt(1-rho^2);
					mu_adj = [mu(j) mu(k)]./sqrt([sig(1,1) sig(2,2)]);
					mu_norm = (mu_adj-rho*fliplr(mu_adj))/rho_adj;
					dsig = [dsigma(j,j),dsigma(j,k);dsigma(k,j),dsigma(k,k)];
					dsig_adj = [dsigma(k,k),-dsigma(j,k);-dsigma(k,j),dsigma(j,j)];

					tr = trace(sig\dsig);
					bvn = bvncdf(mu_adj,rho);

					term1 = mu_adj(1)*normpdf(mu_adj(1))*normcdf(mu_norm(2));
					term2 = mu_adj(2)*normpdf(mu_adj(2))*normcdf(mu_norm(1));
					term3 = normpdf(sqrt(mu_adj(1)^2-2*rho*mu_adj(1)*mu_adj(2)+mu_adj(2)^2)/rho_adj)/sqrt(2*pi);
					moments(1,1) = sig(1,1)*(bvn-term1-rho^2*term2+rho*rho_adj*term3);
					moments(2,2) = sig(2,2)*(bvn-rho^2*term1-term2+rho*rho_adj*term3);
					moments(1,2) = sqrt(sig(1,1)*sig(2,2))*(rho*bvn-rho*term1-rho*term2+rho_adj*term3);
					moments(2,1) = moments(1,2);

					% K is first populated with the derivative of the second raw moment w.r.t. covariance hyperparameters; these derivatives are symmetric
					K(j,k) = -0.5*(tr*bvn+sum(sum((dsig_adj/det(sig)-tr*inv(sig)).*moments)));
					K(k,j) = K(j,k);
				end
			end
			K = K-dM*M'-M*dM';									% Subtract the derivative of pairwise product of the first moments (computed using the product rule)
		elseif i == m+D+1										% Code for computing the derivative of the covariance function w.r.t. likelihood hyperparameters 
			dsigma = 2*sn2*eye(n);								% Compute the derivative of the g-space covariance w.r.t. likelihood hyperparameters 
			dM = probitmean(cov,mn,hyp,x,i);					% Compute the derivative of the f-space mean w.r.t. likelihood hyperparameters 

			K = NaN(n);
			moments = zeros(2);
			for j = 1:n
				for k = 1:j
					sig = [sigma(j,j)+1,sigma(j,k);sigma(k,j),sigma(k,k)+1];	% Compute some relevant quantities; for complete details see Rosenbaum (1961)
					rho = sigma(j,k)/sqrt(sig(1,1)*sig(2,2));
					rho_adj = sqrt(1-rho^2);
					mu_adj = [mu(j) mu(k)]./sqrt([sig(1,1) sig(2,2)]);
					mu_norm = (mu_adj-rho*fliplr(mu_adj))/rho_adj;
					dsig = [dsigma(j,j),dsigma(j,k);dsigma(k,j),dsigma(k,k)];
					dsig_adj = [dsigma(k,k),-dsigma(j,k);-dsigma(k,j),dsigma(j,j)];

					tr = trace(sig\dsig);
					bvn = bvncdf(mu_adj,rho);

					term1 = mu_adj(1)*normpdf(mu_adj(1))*normcdf(mu_norm(2));
					term2 = mu_adj(2)*normpdf(mu_adj(2))*normcdf(mu_norm(1));
					term3 = normpdf(sqrt(mu_adj(1)^2-2*rho*mu_adj(1)*mu_adj(2)+mu_adj(2)^2)/rho_adj)/sqrt(2*pi);
					moments(1,1) = sig(1,1)*(bvn-term1-rho^2*term2+rho*rho_adj*term3);
					moments(2,2) = sig(2,2)*(bvn-rho^2*term1-term2+rho*rho_adj*term3);
					moments(1,2) = sqrt(sig(1,1)*sig(2,2))*(rho*bvn-rho*term1-rho*term2+rho_adj*term3);
					moments(2,1) = moments(1,2);

					% K is first populated with the derivative of the second raw moment w.r.t. covariance hyperparameters; these derivatives are symmetric
					K(j,k) = -0.5*(tr*bvn+sum(sum((dsig_adj/det(sig)-tr*inv(sig)).*moments)));
					K(k,j) = K(j,k);
				end
			end
			K = K-dM*M'-M*dM';									% Subtract the derivative of pairwise product of the first moments (computed using the product rule)
		else 
			error('Unknown hyperparameter for derivative')
	  	end
	end
end

% CDF for the bivariate normal distribution.
function p = bvncdf(b,rho)
	n = size(b,1);
	if rho == 0
		p = cast(prod(Phi(b),2), superiorfloat(b,rho));
	else
		if abs(rho) < 0.3      
			% 6 point Gauss Legendre abscissas and weights
			w = [0.4679139345726904  0.3607615730481384  0.1713244923791705];
			y = [0.2386191860831970  0.6612093864662647  0.9324695142031522];

			% hardcode the doubled and fliplr'd versions for speed
			ww = [0.1713244923791705 0.3607615730481384 0.4679139345726904    ...
				  0.1713244923791705 0.3607615730481384 0.4679139345726904];
			yy = [-0.9324695142031522 -0.6612093864662647 -0.2386191860831970 ...
				  0.9324695142031522 0.6612093864662647 0.2386191860831970];
		elseif abs(rho) < 0.75 
			% 12 point Gauss Legendre abscissas and weights
			w = [0.2491470458134029  0.2334925365383547  0.2031674267230659 ...
				 0.1600783285433464  0.1069393259953183  0.04717533638651177];
			y = [0.1252334085114692  0.3678314989981802  0.5873179542866171 ...
				 0.7699026741943050  0.9041172563704750  0.9815606342467191];

			% hardcode the doubled and fliplr'd versions for speed
			ww = [0.04717533638651177  0.1069393259953183  0.1600783285433464 ...
				  0.2031674267230659   0.2334925365383547  0.2491470458134029 ...
				  0.04717533638651177  0.1069393259953183  0.1600783285433464 ...
				  0.2031674267230659   0.2334925365383547  0.2491470458134029];
			yy = [-0.9815606342467191 -0.9041172563704750 -0.7699026741943050 ...
				  -0.5873179542866171 -0.3678314989981802 -0.1252334085114692 ...
				   0.9815606342467191  0.9041172563704750  0.7699026741943050 ...
				   0.5873179542866171  0.3678314989981802  0.1252334085114692];
		else                 
			% 20 point Gauss Legendre abscissas and weights
			w = [0.1527533871307259  0.1491729864726037  0.1420961093183821  0.1316886384491766  0.1181945319615184 ...
				 0.1019301198172404  0.08327674157670475 0.06267204833410906 0.04060142980038694 0.01761400713915212];
			y = [0.07652652113349733 0.2277858511416451  0.3737060887154196  0.5108670019508271  0.6360536807265150 ...
				 0.7463319064601508  0.8391169718222188  0.9122344282513259  0.9639719272779138  0.9931285991850949];

			% hardcode the doubled and fliplr'd versions for speed
			ww = [0.01761400713915212 0.04060142980038694 0.06267204833410906 0.08327674157670475 0.1019301198172404   ...
				  0.1181945319615184  0.1316886384491766  0.1420961093183821  0.1491729864726037  0.1527533871307259   ...
				  0.01761400713915212 0.04060142980038694 0.06267204833410906 0.08327674157670475 0.1019301198172404   ...
				  0.1181945319615184  0.1316886384491766  0.1420961093183821  0.1491729864726037  0.1527533871307259];
			yy = [-0.9931285991850949 -0.9639719272779138 -0.9122344282513259 -0.8391169718222188 -0.7463319064601508  ...
				  -0.6360536807265150 -0.5108670019508271 -0.3737060887154196 -0.2277858511416451 -0.07652652113349733 ...
				   0.9931285991850949  0.9639719272779138  0.9122344282513259  0.8391169718222188  0.7463319064601508  ...
				   0.6360536807265150  0.5108670019508271  0.3737060887154196  0.2277858511416451  0.07652652113349733 ];
		end

		% For full details on the math implemented below, see Section 2.4 of Genz (2004) 
		% https://link.springer.com/content/pdf/10.1023%2FB%3ASTCO.0000035304.20635.31.pdf
		if abs(rho) < .925
			p1 = prod(Phi(b),2);
			asinrho = asin(rho);
			w = ww;
			theta = asinrho.*(yy+1)/2;
			sintheta = sin(theta);
			cossqtheta = cos(theta).^2; 
			h = -b(:,1); 
			k = -b(:,2);
			hk = h.*k;
			ssq = (h.^2+k.^2)/2;
			p2 = zeros(size(p1));
			for i = 1:n
			  f = exp(-(ssq(i)-hk(i)*sintheta)./cossqtheta);
			  p2(i) = asinrho*sum(w.*f)/2;
			end
			p = p1+p2/(2*pi);
		else 
			if rho > 0
				p1 = Phi(min(b,[],2));
				p1(any(isnan(b),2)) = NaN;
			else
				p1 = Phi(b(:,1))-Phi(-b(:,2));
				p1(p1 < 0) = 0; 
			end

			s = sign(rho);
			if abs(rho) < 1
				h = -b(:,1); 
				k = -b(:,2);
				shk = s.*h.*k;
				asq = 1 - rho.^2; 
				a = sqrt(asq);
				b = abs(h-s.*k); 
				bsq = b.^2;
				c = (4-shk)/8;
				d = c.*(12-shk)/16;

				t1 = a.*(1+d.*asq.^2/5+(c/3-d.*bsq/15).*(asq-bsq));
				t2 = b.*(1-c.*bsq/3+d.*bsq.^2/15);
				p2 = exp(-shk/2).*(t1.*exp(-bsq./(2*asq))-t2.*sqrt(2*pi).*Phi(-b./a));

				w = [w w];
				x = a.*([y-y]+1)/2; 
				x2 = x.^2; 
				x4 = x2.^2;
				sqrt1mx2 = sqrt(1-x2);
				t = (sqrt1mx2-1)./(2*(sqrt1mx2+1));
				for i = 1:n
				  f = exp(-(bsq(i)./x2+shk(i))/2).*(exp(shk(i).*t)./sqrt1mx2-(1+c(i).*x2+d(i).*x4));
				  p2(i) = p2(i)+a.*sum(w.*f)/2;
				end
			else 
				p2 = zeros(class(rho));
			end
			p = p1-s.*p2/(2*pi);
		end
	end
end

% CDF for the univariate normal distribution.
function p = Phi(z)
	p = 0.5*erfc(-z/sqrt(2));
end
