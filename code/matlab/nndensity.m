function [alpha, lambda, dualitygap, objective] = nndensity(Phi, alpha0, verbose)

if nargin < 3
    verbose = false;
end

mu = 10;
eps_feas = 1e-6;
eps0 = 1e-6;
sigma = 0.95;
kappa = 0.01;


if ~(all(alpha0 > 0))
    error('Initial solution not feasible \n')
end

p = numel(alpha0);
diagind = sub2ind([p,p], 1:p, 1:p)';

if p ~= size(Phi, 2)
    error('Length of alpha0 does not match dimension of Phi \n')
end

alpha = alpha0;

convergence = false;

lambda0 = 1./alpha0; %%% surrogate duality gap equal to p = number of constraints
lambda = lambda0;

eta = sum(alpha .* lambda); %%% 'surrogate' duality gap
nu = 0;  %%% is this a good initialization ?

while ~convergence
    gamma = (p * mu)/eta;
    Phialpha = Phi * alpha;
    Dinv = repmat(1./Phialpha, [1, p]);
    tPhiDinv= -(Dinv .* Phi)';
    %%% the different parts of the right hand side
    rdual = sum(tPhiDinv, 2) - lambda + nu;   %%% note the minus sign for lambda -- we have positivity constraints
    rcent = lambda .* alpha - 1/gamma;
    rprimal = sum(alpha) - 1;
    normold = sqrt(sum(rdual.^2 + rcent.^2 + rprimal^2));
    %%%
    Hess = tPhiDinv * tPhiDinv';
    %%% obtain updates for primal variables first
    
    Hess(diagind) = Hess(diagind) + lambda./alpha;
    Hess = [Hess ones(p, 1); [ones(1, p) 0]];
    %diag(Hess) = diag(Hess) + lambda./alpha;
    %%%
    %R = chol(Hess);  %%% hopefully p.d.
    %%%
    rhs_alpha = [-rcent./alpha  - rdual; -rprimal];
    %%%
    delta_alphanu = Hess \ rhs_alpha;
    delta_alpha = delta_alphanu(1:(end-1));
    delta_nu = delta_alphanu(end);
    %%% obtain updates for dual variables
    delta_lambda = (-rcent - lambda .* delta_alpha)./alpha;
    %%%
    %%% line search
    finished = false;
    stepsize = (-lambda ./ delta_lambda); %%% compute maximum stepsize
    if all(stepsize < 0)
       stepsize = 1;
    else
       stepsize = 0.99 * min(stepsize(stepsize > 0));
    end
    
    
    while ~all(alpha + stepsize * delta_alpha > 0)
        stepsize = stepsize * sigma;
    end
    
    while ~finished
        if(verbose)
            disp(['linesearch...' num2str(stepsize)  '\n'])
        end
        Phialphanew = Phi * (alpha + stepsize * delta_alpha);
        Dinvnew = repmat(1./Phialphanew, [1, p]);
        tPhiDinvnew = -(Dinvnew .* Phi)';
        rdual = sum(tPhiDinvnew, 2) - (lambda + stepsize * delta_lambda) + (nu + stepsize * delta_nu);
        rcent =  (lambda + stepsize * delta_lambda).*(alpha + stepsize * delta_alpha) - 1/gamma;
        rprimal = sum(alpha + stepsize * delta_alpha) - 1;
        normnew = sqrt(sum(rdual.^2 + rcent.^2 + rprimal^2));
        if(normnew < (1 - kappa * stepsize) * normold)
            finished = true;
        else
            stepsize = stepsize * sigma;
        end
        %eps
        if(stepsize < (0.5 * eps))
            error('Canceled stepsize selection. \n')
        end
    end
    %%% end of line search
    %%% check convergence and update
    alpha = alpha + stepsize * delta_alpha;
    lambda = lambda + stepsize * delta_lambda;
    nu = nu + stepsize * delta_nu;
    %%% update (surrogate) duality gap
    eta = sum(alpha .* lambda);
    
    if verbose
        disp(['Duality gap: ', num2str(eta), 9])
    end
    
    norm_primal = sqrt(sum(rprimal^2));
    norm_dual = sqrt(sum(rdual.^2));
    
    if verbose
        disp(['norm_primal: ',  num2str(norm_primal), 9])
        disp(['norm_dual:', num2str(norm_dual), 9])
    end
    
    if norm_primal < eps_feas &&  norm_dual < eps_feas && eta < eps0
        convergence = true;
    end
    
    objective = sum(-log(Phi * alpha)); %%% negative log-likelihood
    
    if verbose
        disp(['objective: ', num2str(objective), '\n'])
    end
end

dualitygap = eta;

end

% nndensity <- function(Phi, alpha0, lambda0 = NULL,
%                       mu = 10, eps.feas = 1e-6, eps = 1e-6,
%                       sigma = 0.95, kappa = 0.01, trace = TRUE){
% 
%   #require(Matrix)
%   if(!all(c(alpha0, lambda0) > 0))
%     stop("Initial solution not feasible \n")
% 
%   p <- length(alpha0)
%   if(p != ncol(Phi)) stop("Length of 'alpha0' does not match dimension of 'Phi' \n")
%   alpha <- alpha0 
%   
%   convergence <- FALSE
% 
%   if(is.null(lambda0)) lambda0 <- 1/alpha0 ### -> surrogate duality gap equal to p = number of constraints
% 
%   lambda <- lambda0
%   
%   eta <- sum(alpha * lambda) ### 'surrogate' duality gap
%   nu <- 0  ### is this a good initialization ? 
%   
%   while(!convergence){
%      gamma <- (p * mu)/eta
%      Phialpha <- as.numeric(Phi %*% alpha)
%      Dinv <- Diagonal(x = 1/Phialpha)
%      tPhiDinv <- -Matrix:::t(Phi) %*% Dinv
%      ### the different parts of the right hand side
%      rdual <- Matrix:::rowSums(tPhiDinv) - lambda + rep(nu, p)    ### note the minus sign for lambda -- we have positivity constraints
%      rcent <- lambda * alpha - 1/gamma
%      rprimal <- sum(alpha) - 1
%      normold <- sqrt(sum(rdual^2 + rcent^2 + rprimal^2))
%      ### 
%      Hess <- tPhiDinv %*% Matrix:::t(tPhiDinv) 
%      ### obtain updates for primal variables first
%      Matrix:::diag(Hess) <- Matrix:::diag(Hess) + lambda/alpha
%      Hess <- forceSymmetric(rbind2(cbind2(Hess, rep(1,p)), c(rep(1,p),0)))
%                                     ### note: symmetry enforced 
%      ###
%      R <- Cholesky(Hess) ### hopefully p.d.
%      ###
%      rhs.alpha <- c(-rcent/alpha  - rdual, -rprimal)
%      ###
%      delta.alphanu <- Matrix:::solve(R, rhs.alpha)
%      delta.alpha <- delta.alphanu[-length(delta.alphanu)]
%      delta.nu <- delta.alphanu[length(delta.alphanu)]
%      ### obtain updates for dual variables
%      delta.lambda <- (-rcent - lambda * delta.alpha)/alpha
%      ###
%      ### line search 
%      finished <- FALSE
%      stepsize <- (-lambda / delta.lambda) ### compute maximum stepsize
%      stepsize <- 0.99 * min(stepsize[stepsize > 0])
%      while(!all(alpha + stepsize * delta.alpha > 0))
%        stepsize <- stepsize * sigma 
%      while(!finished){
%        if(trace) cat("linesearch...", stepsize, "\n")
% 
%        Phialphanew <- as.numeric(Phi %*% (alpha + stepsize * delta.alpha))
%        Dinvnew <- Diagonal(x = 1/Phialphanew)
%       tPhiDinvnew <- -Matrix:::t(Phi) %*% Dinvnew
%        rdual <- Matrix:::rowSums(tPhiDinvnew) - (lambda + stepsize * delta.lambda) + rep(nu + stepsize * delta.nu, p) 
%        rcent <-  (lambda + stepsize * delta.lambda)*(alpha + stepsize * delta.alpha) - 1/gamma
%        rprimal <- sum(alpha + stepsize * delta.alpha) - 1
%        normnew <- sqrt(sum(rdual^2 + rcent^2 + rprimal^2))
%        if(normnew < (1 - kappa * stepsize) * normold) finished <- TRUE
%        else stepsize <- stepsize * sigma
%        if(stepsize < (0.5 * .Machine$double.eps)){
%        stop("Canceled stepsize selection. \n")
%      }
%     }
%      ### end of line search
%      ### check convergence and update
%      alpha <- alpha + stepsize * delta.alpha
%      lambda <- lambda + stepsize * delta.lambda
%      nu <- nu + stepsize * delta.nu
%      ### update (surrogate) duality gap
%      eta <- sum(alpha * lambda)
%      if(trace) cat("Duality gap: ", eta, "\t")
%      norm.primal <- sqrt(sum(rprimal^2))
%      norm.dual <- sqrt(sum(rdual^2))
%      cat("norm.primal: ",  norm.primal, "\t")
%      cat("norm.dual:", norm.dual, "\t")
%      if(norm.primal < eps.feas &  norm.dual < eps.feas & eta < eps) convergence <- TRUE
%      objective <- sum(-log(as.numeric(Phi %*% alpha))) ### negative log-likelihood
%      if(trace) cat("objective:", objective, "\n")
%    }
% 
%     return(list(alpha = alpha,
%                 lambda = lambda,
%                 dualitygap = eta,
%                 objective = objective))
% }