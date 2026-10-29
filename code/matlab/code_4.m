%=========================================================================%
% 2D Uncoupled Linear Regression
% Parametric Convergence Rate Test: 2D Bures-Wasserstein Map
%=========================================================================%
clear; clc; close all; rng(2026);

warning('off', 'MATLAB:nearlySingularMatrix');

% =========================================================================
% 1. SIMULATION PARAMETERS
% =========================================================================
sigma_true = 0.5;    % True noise scale
b0_true = [0.5; -0.5]; 

% For the BW map to converge to the true map, the true coefficient matrix
% MUST be Symmetric Positive-Definite (No Rotation).
A_true = [2.0, 0.8; 
          0.8, 1.5]; 

ns = 2.^(6:14);      % Sample sizes: 64 to 16,384
n_mc = 30;           % Monte Carlo iterations per sample size

mse_bw_2d = zeros(length(ns), 1);

% Design generation: Uniform mean 0, variance 1
xgen = @(m) sqrt(3) * (2 * rand(2, m) - 1); 

fprintf('=== Testing Parametric Rate for 2D Bures-Wasserstein Map ===\n');
wb = waitbar(0, 'Running 2D Parametric Rate Sweep...');

% =========================================================================
% 2. MAIN SIMULATION LOOP
% =========================================================================
for i = 1:length(ns)
    n = ns(i);
    errs = zeros(n_mc, 1);
    
    for mc = 1:n_mc
        % Generate Fully Independent Uncoupled Data
        X_observed = xgen(n);
        X_latent   = xgen(n); 
        
        Theta = A_true * X_latent + b0_true;
        Y_observed = Theta + sigma_true * randn(2, n);
        
        % Sample Moments
        mu_x = mean(X_observed, 2); Sigma_x = cov(X_observed');
        mu_y = mean(Y_observed, 2); Sigma_y = cov(Y_observed');
        
        % Denoise covariance and ensure PSD
        Omega_hat = Sigma_y - (sigma_true^2 * eye(2));
        [V, D] = eig(Omega_hat);
        Omega_hat = V * max(D, 0) * V'; 
        
        % 2D Bures-Wasserstein Stretch Operator
        Sx_inv_half = sqrtm(inv(Sigma_x));
        Sx_half     = sqrtm(Sigma_x);
        M_bw = Sx_inv_half * sqrtm(Sx_half * Omega_hat * Sx_half) * Sx_inv_half;
        
        % True map vs Estimated BW map
        Theta_hat_bw = M_bw * (X_observed - mu_x) + mu_y;
        Theta_true   = A_true * X_observed + b0_true;
        
        % Mean Squared Error across all coordinates
        errs(mc) = mean(sum((Theta_hat_bw - Theta_true).^2, 1));
    end
    
    mse_bw_2d(i) = mean(errs);
    waitbar(i/length(ns), wb);
end
close(wb);

% =========================================================================
% 3. EXTRACT EMPIRICAL SLOPE
% =========================================================================
p = polyfit(log(ns), log(mse_bw_2d'), 1);
empirical_slope = p(1);

fprintf('Theoretical Slope: -1.000\n');
fprintf('Empirical Slope:   %.3f\n', empirical_slope);

% =========================================================================
% 4. VISUALIZATION
% =========================================================================
fig = figure('Name', '2D BW Map Parametric Rate', 'Position', [200, 200, 750, 500], 'Color', 'w');
ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); grid(ax, 'minor');

% Plot empirical MSE
loglog(ax, ns, mse_bw_2d, 'b-s', 'LineWidth', 2, 'MarkerSize', 8, ...
       'MarkerFaceColor', 'b', 'DisplayName', sprintf('2D BW Empirical MSE (Slope: %.2f)', empirical_slope));

% Plot theoretical O(1/n) reference line anchored to the first data point
ref_line = mse_bw_2d(1) * (ns(1) ./ ns); 
loglog(ax, ns, ref_line, 'k--', 'LineWidth', 2, 'DisplayName', '$\mathcal{O}(n^{-1})$ Reference');

% UI Display Fixes
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel(ax, 'Sample Size ($n$)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
ylabel(ax, 'Mean Squared Error ($\mathcal{L}_2$ Loss)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
title(ax, '2D Bures-Wasserstein Map: Parametric Convergence Rate', 'Interpreter', 'latex', 'FontSize', 16, 'Color', 'k');

lgd = legend(ax, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);
set(lgd, 'TextColor', 'k', 'Color', 'w');s