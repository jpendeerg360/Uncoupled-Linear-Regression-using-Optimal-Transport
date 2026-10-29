%=========================================================================%
%  Uncoupled LINEAR regression via variance-adjusted empirical Bayes
%-------------------------------------------------------------------------%
%  Tests "Point 1 (EB route)" of the email: recover a linear map
%
%        Y =D Theta + eps,   Theta = beta*X + b,   eps ~ N(0,sigma^2)
%
%  from the *uncoupled* marginals of X and Y (correspondence destroyed).
%
%  Three sources, one idea -- a linear map is fixed by 2nd moments, so the
%  right correction is a *variance adjustment* Omega_hat = Var(Y) - sigma^2:
%    * LinearMap.pdf : OT map T(x)=E[Theta]+ t^Omega_{Sigma_X}(x-mu_X);
%                      parametric rate O_P(1/n).  (1-D: slope = sqrt(Omega/Sigma_X))
%    * Jaffe-Ignatiadis-Sen 2025 (VCB, Thm 3.3 / Alg.1): affine rescaling of
%                      the Bayes/EB denoiser so Cov(delta)=Cov(Theta).
%    * The .m code   : VCB block (sqrt(A/M) rescale) -- but only ever run on
%                      the ISOTONIC sqrt function.  This script runs the
%                      missing LINEAR case.
%
%  Estimators compared:
%    (0) coupled OLS oracle              -- knows the correspondence (benchmark)
%    (1) sort raw shuffled y             -- naive; biased by noise convolution
%    (2) KDE/Tweedie EB denoiser + sort  -- plain EB; OVERSHRINKS
%    (3) VCB variance correction + sort  -- Jaffe Alg.1
%    (4) moment / Bures-Wasserstein plug-in (LinearMap.pdf)
%
%  No toolboxes required (local Gaussian pdf).  CVX not needed.
%=========================================================================%

clear; clc; rng(20260616);
phi = @(z) exp(-0.5*z.^2)/sqrt(2*pi);          % standard normal pdf

% --------------------------- problem settings ---------------------------
sigma  = 0.20;        % noise SD (assumed KNOWN in Sections 1-2)
beta   = 2.0;         % true slope  (set beta<0 to SEE identifiability fail)
b0     = 0.0;         % true intercept
design = 'grid';      % 'grid' = quantile grid of U(0,1]; 'rand' = iid U(0,1)

xgen = @(m) (strcmp(design,'grid')).*((1:m)'/m) + ...
            (strcmp(design,'rand')).*sort(rand(m,1));

%% ===================== Section 1: single-n comparison ==================
n     = 2^11;
x     = xgen(n);
theta = beta*x + b0;                  % latent linear signal
y     = theta + sigma*randn(n,1);     % noisy responses
yperm = y(randperm(n)');              % destroy the X-Y correspondence

sx    = sort(x);
truth = beta*sx + b0;                 % regression fn at sorted x  (the estimand)
mse   = @(f) mean((f - truth).^2);

% (0) coupled OLS oracle (knows correspondence)
coef     = [ones(n,1) x] \ y;
ols_pred = [ones(n,1) sx]*coef;

% (1) naive sort of raw shuffled y
sort_raw = sort(yperm);

% (2) plain EB denoiser (KDE / Tweedie) then sort
dB      = kde_tweedie(yperm, sigma, phi);
sort_eb = sort(dB);

% (3) VCB variance correction (Jaffe-Ignatiadis-Sen 2025, Alg.1) then sort
A_hat    = max(var(yperm) - sigma^2, 0);     % Var(theta) = Var(Y) - sigma^2
M_hat    = var(dB);
t_vcb    = sqrt(A_hat / M_hat);              % affine unshrink factor
mu_hat   = mean(yperm);
dVCB     = t_vcb*(dB - mu_hat) + mu_hat;
sort_vcb = sort(dVCB);

% (4) moment / Bures-Wasserstein 1-D plug-in  (LinearMap.pdf)
beta_hat = sqrt(max(var(yperm)-sigma^2,0) / var(x));   % = sqrt(Omega/Sigma_X)
T_bw     = mu_hat + beta_hat*(sx - mean(x));

% (5) NPMLE / Kiefer-Wolfowitz + Optimal Transport (uir_1d.m)
[~, npmle_pred] = uir_1d(sx, sort(yperm), sigma);

fprintf('\n=== Single-n results (n=%d, beta=%.2f, sigma=%.2f) ===\n', n, beta, sigma);
fprintf('  beta_hat (variance-adjusted) = %.4f   (true %.2f)\n', beta_hat, beta);
fprintf('  MSE  coupled OLS oracle      = %.3e\n', mse(ols_pred));
fprintf('  MSE  sort raw y              = %.3e\n', mse(sort_raw));
fprintf('  MSE  EB (KDE) + sort         = %.3e\n', mse(sort_eb));
fprintf('  MSE  VCB + sort              = %.3e\n', mse(sort_vcb));
fprintf('  MSE  Bures-Wasserstein map   = %.3e\n', mse(T_bw));
fprintf('  MSE  NPMLE + OT map          = %.3e\n', mse(npmle_pred));

figure; hold on; box on;
plot(sx, truth,      'g-',  'LineWidth', 2);
plot(sx, sort_raw,   'Color',[.6 .6 .6]);
plot(sx, sort_eb,    'm-');
plot(sx, sort_vcb,   'b-');
plot(sx, T_bw,       'r--', 'LineWidth', 1.5);
plot(sx, npmle_pred, 'Color',[1 0.5 0], 'LineWidth', 1.5); % Orange line
legend('true beta*x','sort raw','EB+sort (overshrinks)','VCB+sort', ...
       'BW moment map', 'NPMLE + OT map', 'Location','northwest');
xlabel('x (sorted)'); ylabel('fitted value'); title('Linear uncoupled regression: estimators');

%% ===================== Section 2: parametric-rate check ================
% LinearMap.pdf predicts MSE = O_P(1/n).  Confirm the log-log slope ~ -1.
ns        = 2.^(7:13);
reps      = 30;
n_max_kde = 2^13;                 % O(n^2) KDE memory cap
MSErate   = nan(numel(ns), 3);    % columns: [sort_raw, VCB, BW]

for k = 1:numel(ns)
    nk  = ns(k);
    acc = zeros(1,3); cnt = zeros(1,3);
    for r = 1:reps
        xk  = xgen(nk);
        thk = beta*xk + b0;
        yk  = thk + sigma*randn(nk,1);
        ypk = yk(randperm(nk)');
        sxk = sort(xk); trk = beta*sxk + b0;

        e1 = mean((sort(ypk)-trk).^2);                  acc(1)=acc(1)+e1; cnt(1)=cnt(1)+1;

        if nk <= n_max_kde
            dBk = kde_tweedie(ypk, sigma, phi);
            Ak  = max(var(ypk)-sigma^2,0); Mk = var(dBk);
            dVk = sqrt(Ak/Mk)*(dBk-mean(ypk))+mean(ypk);
            e2  = mean((sort(dVk)-trk).^2);             acc(2)=acc(2)+e2; cnt(2)=cnt(2)+1;
        end

        bh  = sqrt(max(var(ypk)-sigma^2,0)/var(xk));
        Tk  = mean(ypk)+bh*(sxk-mean(xk));
        e3  = mean((Tk-trk).^2);                        acc(3)=acc(3)+e3; cnt(3)=cnt(3)+1;
    end
    MSErate(k,:) = acc ./ max(cnt,1);
end

slope_vcb = polyfit(log(ns(:)), log(MSErate(:,2)), 1);
slope_bw  = polyfit(log(ns(:)), log(MSErate(:,3)), 1);
fprintf('\n=== Rate check (log-log slope; -1 == parametric) ===\n');
fprintf('  VCB + sort   slope = %.3f\n', slope_vcb(1));
fprintf('  BW moment    slope = %.3f\n', slope_bw(1));

figure; box on;
loglog(ns, MSErate(:,1),'-o', ns, MSErate(:,2),'-s', ns, MSErate(:,3),'-^', ...
       'LineWidth',1.4); hold on;
loglog(ns, MSErate(1,3)*(ns(1)./ns),'--g');     % 1/n reference through 1st BW point
legend('sort raw','VCB + sort','BW moment','1/n reference','Location','southwest');
xlabel('n'); ylabel('MSE'); title('Parametric-rate check (linear case)');

%% ============== Section 3: sensitivity to the noise scale sigma ========
% sigma was KNOWN above. In practice it must be estimated, and for Gaussian
% X and Gaussian noise it is NOT identifiable from 2nd moments alone:
%        Var(Y) = beta^2 Var(X) + sigma^2   (one equation, two unknowns).
% Sweep the ASSUMED sigma to show how sensitive beta_hat and the MSE are.
n     = 2^12;  x = xgen(n);
theta = beta*x + b0;  y = theta + sigma*randn(n,1);
yperm = y(randperm(n)'); sx = sort(x); truth = beta*sx + b0;

sig_grid   = linspace(0, 0.95*std(yperm), 50);
beta_grid  = sqrt(max(var(yperm)-sig_grid.^2,0)/var(x));
mse_grid   = arrayfun(@(s,bh) mean((mean(yperm)+bh*(sx-mean(x))-truth).^2), ...
                      sig_grid, beta_grid);

figure;
yyaxis left;  plot(sig_grid, beta_grid,'-','LineWidth',1.5); ylabel('\beta estimate');
yyaxis right; plot(sig_grid, mse_grid,'-','LineWidth',1.5);  ylabel('MSE of BW map');
hold on; xline(sigma,'--','true \sigma'); yline(0); 
xlabel('assumed \sigma'); title('Sensitivity to the (unknown) noise scale');

% NOTE on identifiability: with beta<0 the variance-ratio estimator still
% returns |beta| and the monotone sort assumes an increasing map -- the sign
% is unrecoverable from the marginals (OT picks the symmetric-PSD root).

%% ============================ local functions =========================
function d = kde_tweedie(yp, sigma, phi)
% Brown-Greenshtein (2009) KDE empirical-Bayes denoiser via Tweedie:
%   delta(y) = y + tau2 * f'(y)/f(y),  tau2 = sigma^2 + h^2 (effective var).
    nn  = numel(yp);
    h   = std(yp) * nn^(-1/5);          % Silverman bandwidth
    tau2= sigma^2 + h^2;
    D   = yp - yp.';                     % n x n   (O(n^2) memory)
    K   = phi(D/h)/h;
    f   = mean(K, 2);
    df  = mean((-D/h^2).*K, 2);
    d   = yp + tau2*(df./f);
end
