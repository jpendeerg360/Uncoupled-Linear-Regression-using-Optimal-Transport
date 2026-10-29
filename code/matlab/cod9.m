%=========================================================================%
% Multidimensional Confidence Regions (Section 6.4)
% Convergence Rate Test: Joint Parameter Vector theta_d = (b0, vec(S))
%=========================================================================%
clear; clc; close all; rng(2026);

warning('off', 'MATLAB:nearlySingularMatrix');

% =========================================================================
% 1. SIMULATION PARAMETERS
% =========================================================================
d  = 2;
df = d + d^2;               % 6-D parameter vector: 2 intercept + 4 matrix

ns   = 2.^(6:14);           % Sample sizes: 64 to 16,384
n_mc = 60;                   % Monte Carlo repetitions per sample size

sigma_true = 0.5;
b0_true = [1.0; -0.5];

% True coefficient matrix must be SPD (no rotation) so the plain BW map
% recovers it directly -- isolates the rate of theta_d itself, matching
% the identifiability condition used in code6.m.
A_true = [2.0, 0.8;
          0.8, 1.5];

theta_true = [b0_true; A_true(:)]; % 6x1 True Parameter Vector

% Skewed generator (Exponential) so the same estimator machinery as
% code6.m applies unchanged; skewness is irrelevant to this MSE check
% but keeps the data-generating process consistent across the section.
xgen = @(dim, m) exprnd(1, dim, m);

mse_theta = zeros(length(ns), 1);

fprintf('=== Section 6.4: Multivariate Confidence Regions ===\n');
fprintf('Testing Convergence Rate of the 6-D Parameter Estimator theta_hat_d\n');

wb = waitbar(0, 'Running Multivariate Parameter Rate Sweep...');

% =========================================================================
% 2. MAIN SIMULATION LOOP
% =========================================================================
for i = 1:length(ns)
    n = ns(i);
    errs = zeros(n_mc, 1);

    for mc = 1:n_mc
        % -- Fully uncoupled (independent) data generation --
        X_obs = xgen(d, n);
        X_lat = xgen(d, n);
        Theta = A_true * X_lat + b0_true;
        Y_obs = Theta + sigma_true * randn(d, n);

        mu_X = mean(X_obs, 2); Sigma_X = cov(X_obs');
        mu_Y = mean(Y_obs, 2); Sigma_Y = cov(Y_obs');

        M_hat_vec = [mu_X; mu_Y; Sigma_X(:); Sigma_Y(:)];
        theta_hat = moments_to_theta(M_hat_vec, sigma_true^2);

        errs(mc) = sum((theta_hat - theta_true).^2);
    end

    mse_theta(i) = mean(errs);
    waitbar(i/length(ns), wb);
end
close(wb);

% =========================================================================
% 3. EXTRACT EMPIRICAL SLOPE
% =========================================================================
p = polyfit(log(ns), log(mse_theta'), 1);
empirical_slope = p(1);

fprintf('Theoretical Slope: -1.000\n');
fprintf('Empirical Slope:   %.3f\n', empirical_slope);

% =========================================================================
% 4. VISUALIZATION
% =========================================================================
fig = figure('Name', 'Multivariate Parameter Vector Rate', 'Position', [200, 200, 750, 500], 'Color', 'w');
ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); grid(ax, 'minor');

loglog(ax, ns, mse_theta, 'm-d', 'LineWidth', 2, 'MarkerSize', 8, ...
       'MarkerFaceColor', 'm', 'DisplayName', sprintf('$\\hat{\\theta}_d$ MSE (Slope: %.2f)', empirical_slope));

ref_line = mse_theta(1) * (ns(1) ./ ns);
loglog(ax, ns, ref_line, 'k--', 'LineWidth', 2, 'DisplayName', '$\mathcal{O}(n^{-1})$ Reference');

set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel(ax, 'Sample Size ($n$)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
ylabel(ax, 'MSE of $\hat{\theta}_d$ (6-D)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
title(ax, sprintf('Joint Parameter Estimator $\\hat{\\theta}_d$: Convergence Rate ($df=%d$)', df), ...
      'Interpreter', 'latex', 'FontSize', 15, 'Color', 'k');

lgd = legend(ax, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);
set(lgd, 'TextColor', 'k', 'Color', 'w');


% =========================================================================
% HELPER FUNCTION
% =========================================================================
function theta = moments_to_theta(M_vec, sigma2)
    % Maps the 12x1 moment vector to the 6x1 transport map parameter
    % vector theta_d = (b0, vec(S)), identical to the mapping g(M_d) used
    % in code6.m for the Wald ellipsoid coverage test (Section 6.4).
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
    theta = [b0; M_bw(:)];
end