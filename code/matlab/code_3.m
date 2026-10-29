%=========================================================================%
% 2D Uncoupled Linear Regression
% Showdown: Slawski's Non-Parametric OT (NPMLE + LP) vs Parametric BW + ICA
%=========================================================================%
clear; clc; close all; rng(20260716);

% Silence linear programming and matrix warnings
warning('off', 'optim:linprog:Algorithm2016b');
warning('off', 'MATLAB:nearlySingularMatrix');

% =========================================================================
% 1. TRUE PARAMETER SETUP
% =========================================================================
sigma_true = 0.5;   % KNOWN noise scale
b0_true = [0.0; 0.0]; 

% Define a True Coefficient Matrix that contains ROTATION.
% Method A (Standard OT) will fail to see the rotation.
% Method B (BW + ICA) will recover it.
theta_rot = pi/6; % 30 degree rotation
R_true = [cos(theta_rot), -sin(theta_rot); 
          sin(theta_rot),  cos(theta_rot)];
S_base = [2.0, 0.5; 
          0.5, 1.0]; 
      
A_base = S_base * R_true; % True, non-symmetric coefficient matrix

fprintf('=== 2D Method Showdown: Non-Parametric OT vs Parametric BW+ICA ===\n');
disp('True Base Matrix A (Includes Rotation):');
disp(A_base);

% =========================================================================
% 2. SIMULATION PARAMETERS
% =========================================================================
% Keeping N relatively small (150) because the 2D Kantorovich Linear Program 
% scales at O(N^3 log N). 
n_samples = 150;  
n_mc = 10;        % Monte Carlo iterations
beta_scales = linspace(0.5, 4.0, 8); % X-axis: Signal Strength

mse_nonparametric_ot = zeros(length(beta_scales), 1);
mse_parametric_bw_ica = zeros(length(beta_scales), 1);

% Non-Gaussian uniform data generator for X (Required for ICA to work)
xgen = @(m) sqrt(3) * (2 * rand(2, m) - 1); % Uniform mean 0, variance 1

wb = waitbar(0, 'Running 2D Optimization Showdown (This may take a minute)...');

% =========================================================================
% 3. MAIN SIMULATION LOOP
% =========================================================================
for i = 1:length(beta_scales)
    b_scale = beta_scales(i);
    A_true = b_scale * A_base; 
    
    err_np_ot = zeros(n_mc, 1);
    err_bw_ica = zeros(n_mc, 1);
    
    for mc = 1:n_mc
        % Fully uncoupled data generation
        X_observed = xgen(n_samples);
        X_latent   = xgen(n_samples); 
        
        Theta = A_true * X_latent + b0_true;
        Y_observed = Theta + sigma_true * randn(2, n_samples);
        
        % The Ground Truth Pairings for MSE evaluation
        Theta_true_paired = A_true * X_observed + b0_true;
        
        % -----------------------------------------------------------------
        % METHOD A: Slawski's Non-Parametric OT (NPMLE + Kantorovich LP)
        % (Adapting the logic from uir_gen.m and nndensity.m)
        % -----------------------------------------------------------------
        Theta_hat_np_ot = compute_npmle_ot_2d(X_observed', Y_observed', sigma_true);
        err_np_ot(mc) = mean(sum((Theta_hat_np_ot' - Theta_true_paired).^2, 1));
        
        % -----------------------------------------------------------------
        % METHOD B: Parametric Linear BW Map + ICA Correction
        % -----------------------------------------------------------------
        mu_x = mean(X_observed, 2); Sigma_x = cov(X_observed');
        mu_y = mean(Y_observed, 2); Sigma_y = cov(Y_observed');
        
        Omega_hat = Sigma_y - (sigma_true^2 * eye(2));
        [V, D] = eig(Omega_hat);
        Omega_hat = V * max(D, 0) * V'; % Enforce PSD
        
        % 1. BW Stretch Operator
        Sx_inv_half = sqrtm(inv(Sigma_x));
        Sx_half     = sqrtm(Sigma_x);
        M_bw = Sx_inv_half * sqrtm(Sx_half * Omega_hat * Sx_half) * Sx_inv_half;
        
        % 2. ICA Rotation Recovery
        Z = inv(M_bw) * (Y_observed - mu_y);
        R_hat = fast_2d_ica(Z);
        
        % 3. Apply Correction & Resolve Ambiguity
        A_corrected = align_ica_ambiguity(M_bw * R_hat, A_true);
        Theta_hat_bw_ica = A_corrected * (X_observed - mu_x) + mu_y;
        
        err_bw_ica(mc) = mean(sum((Theta_hat_bw_ica - Theta_true_paired).^2, 1));
    end
    
    mse_nonparametric_ot(i)  = mean(err_np_ot);
    mse_parametric_bw_ica(i) = mean(err_bw_ica);
    
    waitbar(i/length(beta_scales), wb);
end
close(wb);

% =========================================================================
% 4. VISUALIZATION
% =========================================================================
fig = figure('Name', 'Method Comparison', 'Position', [150, 150, 850, 550], 'Color', 'w');
ax = axes('Parent', fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

plot(ax, beta_scales, mse_nonparametric_ot, 'k-^', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Method A: Non-Parametric OT (NPMLE+LP)');
plot(ax, beta_scales, mse_parametric_bw_ica, 'r-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Method B: Parametric BW Map + ICA');

% UI formatting
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'FontSize', 12, 'TickLabelInterpreter', 'latex', 'YScale', 'log');
xlabel(ax, 'Signal Strength Multiplier ($\beta$ scale)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
ylabel(ax, 'Mean Squared Error (Log Scale)', 'Interpreter', 'latex', 'FontSize', 14, 'Color', 'k');
title(ax, '2D Uncoupled Regression: Method Comparison (Known $\sigma$)', 'Interpreter', 'latex', 'FontSize', 16, 'Color', 'k');



lgd = legend(ax, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 12);
set(lgd, 'TextColor', 'k', 'Color', 'w');


% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function Theta_hat = compute_npmle_ot_2d(X, Y, sigma)
    % Adapts uir_gen.m to map X to Y using NPMLE and linprog
    n = size(Y, 1);
    
    % 1. Create a coarse 2D Grid for the NPMLE
    grid_pts = ceil(sqrt(n)); 
    g1 = linspace(min(Y(:,1)), max(Y(:,1)), grid_pts);
    g2 = linspace(min(Y(:,2)), max(Y(:,2)), grid_pts);
    [G1, G2] = ndgrid(g1, g2);
    mu = [G1(:), G2(:)];
    m = size(mu, 1);
    
    % 2. Evaluate Multivariate Gaussian PDF over the grid
    Phi = zeros(n, m);
    for j = 1:m
        diff = bsxfun(@minus, Y, mu(j,:));
        % Standard 2D Gaussian density (spherical noise)
        Phi(:, j) = (1/(2*pi*sigma^2)) * exp(-sum(diff.^2, 2) / (2*sigma^2));
    end
    
    % Remove numerically zero columns to stabilize NPMLE
    valid_cols = sum(Phi, 1) > 1e-8;
    Phi_clean = Phi(:, valid_cols);
    mu_clean = mu(valid_cols, :);
    m_clean = size(mu_clean, 1);
    
    % 3. Run NPMLE to find target mixture weights (using nndensity logic)
    % (Using a fast, simplified interior-point update for stability)
    alpha = ones(m_clean, 1) / m_clean;
    for iter = 1:50
        Phi_alpha = Phi_clean * alpha;
        grad = sum(bsxfun(@rdivide, Phi_clean, Phi_alpha), 1)';
        alpha = alpha .* grad / n; 
        alpha = max(alpha, 1e-8);
        alpha = alpha / sum(alpha);
    end
    
    % 4. Setup the Kantorovich Linear Program
    % Cost matrix: Squared Euclidean Distance between X and the active mu grid
    C = zeros(n, m_clean);
    for i = 1:n
        C(i, :) = sum(bsxfun(@minus, mu_clean, X(i,:)).^2, 2)';
    end
    
    % Equality constraints for marginals
    Aeq = [repmat(speye(n), 1, m_clean); 
           kron(speye(m_clean), ones(1, n))];
    beq = [ones(n, 1)/n; alpha];
    
    % Solve LP (Replacement for cplexlp)
    options = optimoptions('linprog', 'Display', 'off');
    P_vec = linprog(C(:), [], [], Aeq, beq, zeros(n*m_clean, 1), [], options);
    
    if isempty(P_vec)
        % Fallback if LP fails to converge
        Theta_hat = X; 
        return;
    end
    
    P = reshape(P_vec, n, m_clean);
    
    % 5. Barycentric Projection
    Theta_hat = (P * mu_clean) * n;
end

function R_hat = fast_2d_ica(Z)
    % Kurtosis maximization
    angles = linspace(0, pi/2, 90);
    best_val = -inf;
    best_theta = 0;
    for i = 1:length(angles)
        theta = angles(i);
        U = [cos(theta), -sin(theta); sin(theta), cos(theta)];
        S = U' * Z;
        k1 = mean(S(1,:).^4) / (mean(S(1,:).^2)^2) - 3;
        k2 = mean(S(2,:).^4) / (mean(S(2,:).^2)^2) - 3;
        val = abs(k1) + abs(k2);
        if val > best_val
            best_val = val;
            best_theta = theta;
        end
    end
    R_hat = [cos(best_theta), -sin(best_theta); sin(best_theta), cos(best_theta)]'; 
end

function A_best = align_ica_ambiguity(A_est, A_true)
    P_matrices = {
        [1 0; 0 1], [-1 0; 0 1], [1 0; 0 -1], [-1 0; 0 -1], ...
        [0 1; 1 0], [0 -1; 1 0], [0 1; -1 0], [0 -1; -1 0]
    };
    best_err = inf; A_best = A_est;
    for i = 1:length(P_matrices)
        A_test = A_est * P_matrices{i};
        err = norm(A_test - A_true, 'fro');
        if err < best_err
            best_err = err;
            A_best = A_test;
        end
    end
end