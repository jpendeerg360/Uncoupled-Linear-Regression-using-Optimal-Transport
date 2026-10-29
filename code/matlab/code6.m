%=========================================================================%
% Multidimensional Confidence Regions (Section 6.4)
% Validating the 2D Asymptotic Wald Ellipsoid vs Non-Parametric Bootstrap
%=========================================================================%
clear; clc; close all; rng(2026);

warning('off', 'MATLAB:nearlySingularMatrix');

% =========================================================================
% 1. SIMULATION PARAMETERS
% =========================================================================
n_mc      = 1000;    % Monte Carlo coverage iterations
n_samples = 16384;    % Sample size per iteration
n_boot    = 400;     % Resamples for the Bootstrap baseline
alpha     = 0.05;    % 95% Confidence Level

d = 2;
% Degrees of freedom: 2 intercept params + 3 UNIQUE matrix params (the
% coefficient matrix is symmetric, so its off-diagonal entry is not a
% free parameter -- using d^2 here double-counts M(1,2)=M(2,1) and makes
% Sigma_theta structurally singular). df = d + d*(d+1)/2 = 5.
df = d + d*(d+1)/2;
chi2_crit = chi2inv(1 - alpha, df);

sigma_true = 0.5;
b0_true = [1.0; -0.5]; 

% True coefficient matrix must be Symmetric Positive-Definite (No rotation)
% to isolate and verify the BW map's parametric distribution.
A_true = [2.0, 0.8; 
          0.8, 1.5]; 

% 5x1 True Parameter Vector using the unique (vech) entries of A_true:
% [b0; A(1,1); A(2,1); A(2,2)]. Do NOT use A_true(:), which would repeat
% the off-diagonal entry and make theta's covariance singular.
theta_true = [b0_true; A_true(1,1); A_true(2,1); A_true(2,2)];

% Generator: Exponential distribution (Mean 1, Skewed). 
% We use a skewed distribution on purpose! If we used a symmetric 
% distribution (like Gaussian/Uniform), the 3rd moment blocks (Gamma_3) 
% in V_d would be exactly zero. The exponential distribution forces the 
% simulation to test the full, dense 12x12 asymptotic covariance matrix.
xgen = @(dim, m) exprnd(1, dim, m);

fprintf('=== Section 6.4: Multidimensional Confidence Regions ===\n');
fprintf('Testing 95%% Coverage for %d-D Parameter Vector (N = %d)...\n', df, n_samples);
fprintf('(df = %d uses the vech parameterization; A is symmetric so its\n', df);
fprintf(' off-diagonal entry is one free parameter, not two.)\n\n');

clt_coverage  = 0;
boot_coverage = 0;

wb = waitbar(0, 'Running Multivariate Coverage Tests...');

% =========================================================================
% 2. MAIN MONTE CARLO LOOP
% =========================================================================
for mc = 1:n_mc
    % -- 1. Data Generation (Fully Uncoupled) --
    X_obs  = xgen(d, n_samples);
    X_lat  = xgen(d, n_samples); 
    Theta  = A_true * X_lat + b0_true;
    Y_obs  = Theta + sigma_true * randn(d, n_samples);
    
    mu_X = mean(X_obs, 2); 
    mu_Y = mean(Y_obs, 2);
    
    % Centered data for moment extraction
    Xc = bsxfun(@minus, X_obs, mu_X);
    Yc = bsxfun(@minus, Y_obs, mu_Y);
    
    % -- 2. Assemble Empirical V_d (Eq 40) --
    % We construct the per-sample moment vector: Z = [x; vec(x*x')]
    Z_X = zeros(d + d^2, n_samples);
    Z_Y = zeros(d + d^2, n_samples);
    
    for i = 1:n_samples
        Z_X(:, i) = [X_obs(:, i); reshape(Xc(:, i) * Xc(:, i)', 4, 1)];
        Z_Y(:, i) = [Y_obs(:, i); reshape(Yc(:, i) * Yc(:, i)', 4, 1)];
    end
    
    % V_X and V_Y contain the 1st, 2nd, 3rd, and 4th moment asymptotic variances
    V_X = cov(Z_X'); 
    V_Y = cov(Z_Y'); 
    
    % Block-diagonalize to form the 12x12 V_d matrix (Enforcing Uncoupled Independence)
    % M_vec order: [mu_X (1:2); mu_Y (3:4); vec(Sig_X) (5:8); vec(Sig_Y) (9:12)]
    V_total = zeros(12, 12);
    V_total(1:2, 1:2)   = V_X(1:2, 1:2);       % Sigma_X
    V_total(3:4, 3:4)   = V_Y(1:2, 1:2);       % Sigma_Y
    V_total(5:8, 5:8)   = V_X(3:6, 3:6);       % Psi_{4,X}
    V_total(9:12, 9:12) = V_Y(3:6, 3:6);       % Psi_{4,Y}
    
    % Cross-terms between mean and variance (Gamma_3 blocks)
    V_total(1:2, 5:8)   = V_X(1:2, 3:6);
    V_total(5:8, 1:2)   = V_X(3:6, 1:2);
    V_total(3:4, 9:12)  = V_Y(1:2, 3:6);
    V_total(9:12, 3:4)  = V_Y(3:6, 1:2);
    
    % -- 3. Base Point Estimates --
    Sig_X_hat = reshape(mean(Z_X(3:6, :), 2), 2, 2);
    Sig_Y_hat = reshape(mean(Z_Y(3:6, :), 2), 2, 2);
    M_hat_vec = [mu_X; mu_Y; Sig_X_hat(:); Sig_Y_hat(:)];
    
    theta_hat = moments_to_theta(M_hat_vec, sigma_true^2);
    
    % -- 4. METHOD A: CLT / Delta Method --
    % Compute the 5x12 Jacobian using central finite differences
    J_d = zeros(df, 12);
    eps_fd = 1e-6;
    for k = 1:12
        M_plus = M_hat_vec; M_plus(k) = M_plus(k) + eps_fd;
        M_minus = M_hat_vec; M_minus(k) = M_minus(k) - eps_fd;
        t_plus = moments_to_theta(M_plus, sigma_true^2);
        t_minus = moments_to_theta(M_minus, sigma_true^2);
        J_d(:, k) = (t_plus - t_minus) / (2 * eps_fd);
    end
    
    % Equation 42: Assemble Parameter Covariance
    Sigma_theta_clt = (1/n_samples) * (J_d * V_total * J_d');
    
    % Wald Ellipsoid Check
    diff = theta_hat - theta_true;
    mahal_clt = diff' * (Sigma_theta_clt \ diff);
    if mahal_clt <= chi2_crit
        clt_coverage = clt_coverage + 1;
    end
    
    % -- 5. METHOD B: Non-Parametric Bootstrap --
    theta_boot_reps = zeros(df, n_boot);
    for b = 1:n_boot
        idx_X = randi(n_samples, n_samples, 1);
        idx_Y = randi(n_samples, n_samples, 1);
        
        X_star = X_obs(:, idx_X);
        Y_star = Y_obs(:, idx_Y);
        
        M_star = [mean(X_star, 2); mean(Y_star, 2); reshape(cov(X_star'), 4, 1); reshape(cov(Y_star'), 4, 1)];
        theta_boot_reps(:, b) = moments_to_theta(M_star, sigma_true^2);
    end
    
    Sigma_theta_boot = cov(theta_boot_reps');
    mahal_boot = diff' * (Sigma_theta_boot \ diff);
    if mahal_boot <= chi2_crit
        boot_coverage = boot_coverage + 1;
    end
    
    waitbar(mc/n_mc, wb);
end
close(wb);

cov_clt_pct = (clt_coverage / n_mc) * 100;
cov_boot_pct = (boot_coverage / n_mc) * 100;
fprintf('CLT Wald Ellipsoid Coverage:       %.2f%%\n', cov_clt_pct);
fprintf('Bootstrap Wald Ellipsoid Coverage: %.2f%%\n', cov_boot_pct);

% =========================================================================
% 3. VISUALIZATION
% =========================================================================
fig = figure('Name', 'Multivariate Confidence Regions', 'Position', [200, 200, 700, 500], 'Color', 'w');
ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

bars = bar(ax, [cov_clt_pct, cov_boot_pct], 'FaceColor', 'flat');
bars.CData(1,:) = [0.2 0.6 0.8]; % Blue for CLT
bars.CData(2,:) = [0.8 0.3 0.3]; % Red for Bootstrap

yline(ax, 95, 'g--', 'Nominal 95% Target', 'LineWidth', 2.5, 'LabelHorizontalAlignment', 'center', 'Interpreter', 'latex', 'FontSize', 12);

set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'XTick', 1:2, ...
    'XTickLabel', {'Multivariate CLT (Eq 41)', 'Non-Parametric Bootstrap'}, ...
    'TickLabelInterpreter', 'latex', 'FontSize', 12, 'YLim', [85 100]);
ylabel(ax, 'Empirical Coverage (\%)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
title(ax, sprintf('Joint Parameter Coverage (%d-D Ellipsoid, $N = %d$)', df, n_samples), 'Interpreter', 'latex', 'FontSize', 15, 'Color', 'k');


% =========================================================================
% HELPER FUNCTIONS
% =========================================================================
function theta = moments_to_theta(M_vec, sigma2)
    % Maps the 12x1 moment vector to the 5x1 transport map parameter vector.
    % theta = [b0; M_bw(1,1); M_bw(2,1); M_bw(2,2)] -- the UNIQUE entries
    % of the symmetric M_bw. Using M_bw(:) (all 4 entries) would repeat
    % M_bw(2,1)=M_bw(1,2) exactly, making theta's covariance singular.
    mu_X  = M_vec(1:2);
    mu_Y  = M_vec(3:4);
    Sig_X = reshape(M_vec(5:8), 2, 2);
    Sig_Y = reshape(M_vec(9:12), 2, 2);

    % Symmetrize to prevent numerical precision drift into complex planes
    Sig_X = (Sig_X + Sig_X')/2;
    Sig_Y = (Sig_Y + Sig_Y')/2;

    Omega = Sig_Y - sigma2 * eye(2);
    [V, D] = eig(Omega); 
    Omega = V * max(D, 0) * V'; % Enforce PSD

    Sx_inv_half = real(sqrtm(inv(Sig_X)));
    Sx_half     = real(sqrtm(Sig_X));
    M_bw        = real(Sx_inv_half * sqrtm(Sx_half * Omega * Sx_half) * Sx_inv_half);
    
    % Symmetrize mapping operator
    M_bw = (M_bw + M_bw')/2;

    b0 = mu_Y - M_bw * mu_X;
    theta = [b0; M_bw(1,1); M_bw(2,1); M_bw(2,2)];
end