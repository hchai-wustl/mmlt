% This script is an example of the workflow for "fitting hyperparameters in f-space" 
% as described in Chai (2018)

startup

% setup some problem parameters
C = 2;
xmax = 3;
xmin = -3;
GP_inf = @infExact;
GP_lik = @likGauss;
GP_mean = @meanConst;
GP_cov = {@covMaternard, 3};
params.length = -100;
params.method = 'LBFGS';

% observation locations, sampled uniformly at random over the interval [-2,2]
x = -2+4*rand(5,1);

% test locations, a dense grid over the interval [-3,3]
test_x = (xmin:0.01:xmax)';
num_tests = size(test_x,1);

% compute the observations and all transformations of those observations
y = 0.95*exp(-C*x.^2);
l = log(y);
p = norminv(y);

% the basic workflow for each transformation below is as follows:
% 1. Optimize hyperparameters in f-space
% 2. Using the learned hyperparameters, compute the g-space posterior belief at the test locations
% 3. Transform the g-space posterior belief into an f-space posterior belief 

% no transformation (baseline for comparison)
hyp = struct('mean',mean(y),'cov',[log(0.5) log(1/3)],'lik',log(0.001));
hyp_none = minimize_v2(hyp,@gp,params,GP_inf,GP_mean,GP_cov,GP_lik,x,y);
[mn,vrc] = gp(hyp_none,GP_inf,GP_mean,GP_cov,GP_lik,x,y,test_x);

% log transformation
hyp = struct('mean',mean(l),'cov',[log(0.5) log(0.5)],'lik',log(0.001));
hyp_log = minimize_v2(hyp,@gp,params,@loginfExact,GP_mean,GP_cov,GP_lik,x,l);
[log_mn,log_vrc] = gp(hyp_log,GP_inf,GP_mean,GP_cov,GP_lik,x,l,test_x);
log_est = exp(log_mn+0.5*log_vrc);
log_bounds = zeros(num_tests,2);
log_bounds(:,1) = logninv(0.025,log_mn,sqrt(log_vrc));
log_bounds(:,2) = logninv(0.975,log_mn,sqrt(log_vrc));
log_var = exp(2*log_mn+2*log_vrc)-exp(2*log_mn+log_vrc);

% probit transformation
hyp = struct('mean',mean(p),'cov',[log(0.5) log(0.5)],'lik',log(0.001));
hyp_probit = minimize_v2(hyp,@gp,params,@probitinfExact,GP_mean,GP_cov,GP_lik,x,p);
[probit_mn,probit_vrc] = gp(hyp_probit,GP_inf,GP_mean,GP_cov,GP_lik,x,p,test_x);
probit_est = normcdf(probit_mn./sqrt(1+probit_vrc));
probit_bounds = zeros(num_tests,2);
probit_var = zeros(num_tests,1);
% the credible interval for the probit transformation cannot be computed in closed form
% each location's credible interval is estimated in the for loop below
for i = 1:num_tests
	probit_var(i) = bvncdf(probit_mn(i)/sqrt(probit_vrc(i)+1)*ones(1,2),probit_vrc(i)/(probit_vrc(i)+1))-probit_est(i).^2;
    try
        probit_bounds(i,1) = fzero(@(z) integral(@(x) normpdf(norminv(x),probit_mn(i),sqrt(probit_vrc(i)))./normpdf(norminv(x)),1e-12,z)-0.025,[1e-12,1]);
    catch
        probit_bounds(i,1) = probit_est(i);
    end
    try
        probit_bounds(i,2) = fzero(@(z) integral(@(x) normpdf(norminv(x),probit_mn(i),sqrt(probit_vrc(i)))./normpdf(norminv(x)),1e-12,z)-0.975,[1e-12,1]);
    catch
        probit_bounds(i,2) = probit_est(i);
    end
end

set(0,'DefaultFigureWindowStyle','docked');

figure;
plot(test_x,mn);
hold on;
plot(x,0.95*exp(-C*x.^2),'x')
fill([test_x' fliplr(test_x')],[(mn+2*sqrt(vrc))' fliplr((mn-2*sqrt(vrc))')],1,'facecolor','b','edgecolor','none','facealpha',0.2);
axis([xmin,xmax,-1,1.5]);
hold off;

figure;
plot(test_x,log_est);
hold on;
plot(x,0.95*exp(-C*x.^2),'x')
fill([test_x' fliplr(test_x')],[log_bounds(:,1)' fliplr(log_bounds(:,2)')],1,'facecolor','b','edgecolor','none','facealpha',0.2);
axis([xmin,xmax,-1,1.5]);
hold off;

figure;
plot(test_x,log_est);
hold on;
plot(x, 0.95*exp(-C*x.^2), 'x')
fill([test_x' fliplr(test_x')],[(log_est-2*sqrt(log_var))' fliplr((log_est+2*sqrt(log_var))')],1,'facecolor','b','edgecolor','none','facealpha',0.2)
axis([xmin,xmax,-1,1.5]);
hold off;

figure;
plot(test_x,probit_est);
hold on;
plot(x,0.95*exp(-C*x.^2),'x')
fill([test_x' fliplr(test_x')],[probit_bounds(:,1)' fliplr(probit_bounds(:,2)')],1,'facecolor','b','edgecolor','none','facealpha',0.2);
axis([xmin,xmax,-1,1.5]);
hold off;

figure;
plot(test_x,probit_est);
hold on;
plot(x, 0.95*exp(-C*x.^2), 'x')
fill([test_x' fliplr(test_x')],[(probit_est-2*sqrt(probit_var))' fliplr((probit_est+2*sqrt(probit_var))')],1,'facecolor','b','edgecolor','none','facealpha',0.2)
axis([xmin,xmax,-1,1.5]);
hold off;
