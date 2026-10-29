%=========================================================================%
% Multidimensional Noise Scale Determination (Section 6.3.1)
% Convergence Rate Test: Eigendecomposition Method (Rank-Deficient, d = 3, r = 2)
%=========================================================================%
clear; clc; close all; rng(2026);

warning('off', 'MATLAB:nearlySingularMatrix');

% =========================================================================
% 1. SIMULATION PARAMETERS
% =========================================================================
d = 3;                      % Ambient dimension
r = 2;                       % rank(A) < d, so Omega = A*Sigma_X*A' has a
                              % (d-r)-dimensional null space
sigma_true  = 0.50;
sigma2_true = sigma_true^2;

ns   = 2.^(6:14);            % Sample sizes: 64 to 16,384
n_mc = 60;                    % Monte Carlo repetitions per sample size

mse_eigen = zeros(length(ns), 1);

% Marginal generator: no non-Gaussianity requirement for this method
% (Section 6.3.1 relies purely on second moments / Weyl's inequality),
% so a standard Gaussian design is used here on purpose.
xgen = @(dim, m) randn(dim, m);

% Rank-2 coefficient matrix in 3D: the 3rd dimension carries no signal,
% so Sigma_Y's smallest eigenvalue is exactly sigma^2 in population.
A_rd = [2.0,  0.5,  0.0;
        0.5,  1.5,  0.0;
        0.0,  0.0,  0.0];
b0_true = zeros(d, 1);

fprintf('=== Section 6.3.1: Eigendecomposition Noise Estimation ===\n');
fprintf('Testing Convergence Rate of sigma^2 estimator (d = %d, r = %d)\n', d, r);

wb = waitbar(0, 'Running Eigendecomposition Rate Sweep...');

% =========================================================================
% 2. MAIN SIMULATION LOOP
% =========================================================================
for i = 1:length(ns)
    n = ns(i);
    errs = zeros(n_mc, 1);

    for mc = 1:n_mc
        % -- Fully uncoupled (independent) data generation --
        X_lat = xgen(d, n);
        Theta = A_rd * X_lat + b0_true;
        Y_obs = Theta + sigma_true * randn(d, n);

        sigma2_hat = eigendecomposition_estimate(Y_obs, r);
        errs(mc) = (sigma2_hat - sigma2_true)^2;
    end

    mse_eigen(i) = mean(errs);
    waitbar(i/length(ns), wb);
end
close(wb);

% =========================================================================
% 3. EXTRACT EMPIRICAL SLOPE
% =========================================================================
p = polyfit(log(ns), log(mse_eigen'), 1);
empirical_slope = p(1);

fprintf('Theoretical Slope: -1.000\n');
fprintf('Empirical Slope:   %.3f\n', empirical_slope);

% =========================================================================
% 4. VISUALIZATION
% =========================================================================
fig = figure('Name', 'Eigendecomposition Method Rate', 'Position', [200, 200, 750, 500], 'Color', 'w');
ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); grid(ax, 'minor');

loglog(ax, ns, mse_eigen, 'r-o', 'LineWidth', 2, 'MarkerSize', 8, ...
       'MarkerFaceColor', 'r', 'DisplayName', sprintf('Eigendecomp. $\\hat{\\sigma}^2$ MSE (Slope: %.2f)', empirical_slope));

ref_line = mse_eigen(1) * (ns(1) ./ ns);
loglog(ax, ns, ref_line, 'k--', 'LineWidth', 2, 'DisplayName', '$\mathcal{O}(n^{-1})$ Reference');

set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel(ax, 'Sample Size ($n$)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
ylabel(ax, 'MSE of $\hat{\sigma}^2$', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
title(ax, sprintf('Eigendecomposition Noise Estimator: Convergence Rate ($d=%d$, $r=%d$)', d, r), ...
      'Interpreter', 'latex', 'FontSize', 15, 'Color', 'k');

lgd = legend(ax, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);
set(lgd, 'TextColor', 'k', 'Color', 'w');


% =========================================================================
% HELPER FUNCTION
% =========================================================================
function sigma2_hat = eigendecomposition_estimate(Y, r)
    % Implements Equations (32)-(33): average the smallest (d-r) sample
    % eigenvalues of Sigma_Y to isolate the isotropic noise floor.
    Sigma_Y = cov(Y');
    Sigma_Y = (Sigma_Y + Sigma_Y') / 2; % symmetrize for numerical safety

    evals = sort(eig(Sigma_Y), 'descend');
    d = size(Y, 1);

    sigma2_hat = mean(evals(r+1 : d));
    sigma2_hat = max(sigma2_hat, 0);
end