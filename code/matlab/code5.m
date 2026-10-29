%=========================================================================%
% Multidimensional Noise Scale Determination (Section 6.3)
% Method 1: Eigendecomposition (Rank-Deficient Case)
% Method 2: Directional Cumulants + ICA (Full-Rank Case)
%=========================================================================%
clear; clc; close all; rng(2026);

% Standard non-Gaussian independent generator (Uniform) for ICA/Cumulants
xgen = @(d, m) sqrt(3) * (2 * rand(d, m) - 1); 

n_samples = 5000; % Large N required for stable 4th cumulant estimation
true_sigma = 0.50; 
true_sigma2 = true_sigma^2;

fprintf('=== Section 6.3: Multidimensional Noise Estimation ===\n\n');

%% =======================================================================%
%  METHOD 1: EIGENDECOMPOSITION (Section 6.3.1)
%  Condition: A is rank-deficient (d = 3, r = 2)
% ========================================================================%
fprintf('--- Method 1: Eigendecomposition (Rank-Deficient) ---\n');

d1 = 3; r1 = 2;
X1 = xgen(d1, n_samples);

% Create a Rank-2 Coefficient Matrix in 3D
A_rd = [2.0,  0.5,  0.0; 
        0.5,  1.5,  0.0; 
        0.0,  0.0,  0.0]; % The 3rd dimension has no signal

b0_1 = [0; 0; 0];
Y1 = (A_rd * X1 + b0_1) + true_sigma * randn(d1, n_samples);

% 1. Compute response covariance
Sigma_Y1 = cov(Y1');

% 2. Extract eigenvalues and sort descending
evals1 = sort(eig(Sigma_Y1), 'descend');

% 3. Average the smallest (d - r) eigenvalues (Equation 33)
sigma2_hat_eigen = mean(evals1(r1+1 : d1));
sigma_hat_eigen  = sqrt(max(sigma2_hat_eigen, 0));

fprintf('True Sigma:      %.4f\n', true_sigma);
fprintf('Estimated Sigma: %.4f\n', sigma_hat_eigen);
fprintf('Absolute Error:  %.4e\n\n', abs(true_sigma - sigma_hat_eigen));

%% =======================================================================%
%  METHOD 2: DIRECTIONAL CUMULANTS (Section 6.3.2)
%  Condition: A is full-rank, independent non-Gaussian components (d = 2)
% ========================================================================%
fprintf('--- Method 2: Directional Cumulants + ICA (Full-Rank) ---\n');

d2 = 2;
X2 = xgen(d2, n_samples);

% Create a Full-Rank, Rotated Coefficient Matrix
theta_rot = pi/6; 
R_true = [cos(theta_rot), -sin(theta_rot); 
          sin(theta_rot),  cos(theta_rot)];
S_base = [2.0, 0.5; 
          0.5, 1.0]; 
A_fr = S_base * R_true; 

b0_2 = [0; 0];
Y2 = (A_fr * X2 + b0_2) + true_sigma * randn(d2, n_samples);

% 1. Central Moments
mu_x = mean(X2, 2); Sigma_X2 = cov(X2');
mu_y = mean(Y2, 2); Sigma_Y2 = cov(Y2');

% 2. Get BW Operator (Assuming naive noise=0 for the initial stretch matrix)
Sx_inv_half = sqrtm(inv(Sigma_X2));
Sx_half     = sqrtm(Sigma_X2);
M_bw_naive  = Sx_inv_half * sqrtm(Sx_half * Sigma_Y2 * Sx_half) * Sx_inv_half;

% 3. Extract Rotation via ICA on de-stretched responses
Z = inv(M_bw_naive) * bsxfun(@minus, Y2, mu_y);
R_hat = fast_2d_ica(Z);

% 4. Directional Cumulant Projections (Equations 37 & 38)
[U, evals2_mat] = eig(Sigma_Y2); % U contains eigenvectors of Sigma_Y
sigma2_directional = zeros(d2, 1);

for j = 1:d2
    u_j = U(:, j);               % Eigenvector in Y-space
    w_j = R_hat' * u_j;          % Rotated corresponding vector in X-space
    
    % Project the data
    Y_proj = u_j' * Y2; 
    X_proj = w_j' * X2;
    
    % 1D Variances
    var_Y_proj = var(Y_proj, 1);
    var_X_proj = var(X_proj, 1);
    
    % 1D 4th Cumulants
    k4_Y_proj = mean((Y_proj - mean(Y_proj)).^4) - 3 * var_Y_proj^2;
    k4_X_proj = mean((X_proj - mean(X_proj)).^4) - 3 * var_X_proj^2;
    
    % 1D Cumulant Noise Estimate along this basis direction
    if abs(k4_X_proj) > 1e-10 && sign(k4_X_proj) == sign(k4_Y_proj)
        ratio = sqrt(k4_Y_proj / k4_X_proj);
        sigma2_directional(j) = var_Y_proj - var_X_proj * ratio;
    else
        sigma2_directional(j) = 0; % Fallback if cumulants collapse
    end
end

% 5. Pool the directional estimates (Equation 38)
sigma2_hat_cumulant = mean(sigma2_directional);
sigma_hat_cumulant  = sqrt(max(sigma2_hat_cumulant, 0));

fprintf('True Sigma:      %.4f\n', true_sigma);
fprintf('Estimated Sigma: %.4f\n', sigma_hat_cumulant);
fprintf('Absolute Error:  %.4e\n\n', abs(true_sigma - sigma_hat_cumulant));

%% =======================================================================%
%  VISUALIZATION: The Mechanisms of Noise Estimation
% ========================================================================%
fig = figure('Name', 'Multidimensional Noise Estimation', 'Position', [150, 150, 950, 450], 'Color', 'w');

% --- Subplot 1: Eigendecomposition (Method 1) ---
ax1 = subplot(1, 2, 1); hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on');

% Convert eigenvalues to standard deviation scale for plotting
sqrt_evals = sqrt(evals1); 
b1 = bar(ax1, 1:d1, sqrt_evals, 'FaceColor', 'flat');
b1.CData(1:2,:) = repmat([0.2 0.6 0.8], 2, 1); % Blue for signal components
b1.CData(3,:)   = [0.8 0.3 0.3];               % Red for the null-space (noise) component

yline(ax1, true_sigma, 'k--', 'True $\sigma$', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left', 'Interpreter', 'latex', 'FontSize', 12);

set(ax1, 'XTick', 1:d1, 'XTickLabel', {'$\sqrt{\lambda_1}$', '$\sqrt{\lambda_2}$', '$\sqrt{\lambda_3}$ (Null Space)'}, ...
    'TickLabelInterpreter', 'latex', 'FontSize', 12, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
ylabel(ax1, 'Standard Deviation Scale', 'Interpreter', 'latex', 'FontSize', 14);
title(ax1, 'Method 1: Null Space Isolation (3D)', 'Interpreter', 'latex', 'FontSize', 15);


% --- Subplot 2: Directional Cumulants (Method 2) ---
ax2 = subplot(1, 2, 2); hold(ax2, 'on'); box(ax2, 'on'); grid(ax2, 'on');

% Gather estimates (converted to standard deviations)
sigma_directional_estimates = sqrt(max(sigma2_directional, 0));
estimates_to_plot = [sigma_directional_estimates; sigma_hat_cumulant];

b2 = bar(ax2, 1:3, estimates_to_plot, 'FaceColor', 'flat');
b2.CData(1:2,:) = repmat([0.2 0.8 0.4], 2, 1); % Green for directional estimates
b2.CData(3,:)   = [0.8 0.3 0.3];               % Red for the pooled final estimate

yline(ax2, true_sigma, 'k--', 'True $\sigma$', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left', 'Interpreter', 'latex', 'FontSize', 12);

set(ax2, 'XTick', 1:3, 'XTickLabel', {'Direction 1 ($\hat{u}_1$)', 'Direction 2 ($\hat{u}_2$)', 'Pooled Estimate'}, ...
    'TickLabelInterpreter', 'latex', 'FontSize', 12, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'YLim', [0, true_sigma * 1.5]);
ylabel(ax2, 'Estimated Noise Scale ($\hat{\sigma}$)', 'Interpreter', 'latex', 'FontSize', 14);
title(ax2, 'Method 2: Directional Cumulants (2D)', 'Interpreter', 'latex', 'FontSize', 15);


%% =======================================================================%
%  HELPER FUNCTION: 2D Kurtosis-Maximization ICA
% ========================================================================%
function R_hat = fast_2d_ica(Z)
    angles = linspace(0, pi/2, 90);
    best_val = -inf;
    best_theta = 0;
    
    for i = 1:length(angles)
        theta = angles(i);
        U_rot = [cos(theta), -sin(theta); 
                 sin(theta),  cos(theta)];
        S = U_rot' * Z;
        
        k1 = mean(S(1,:).^4) / (mean(S(1,:).^2)^2) - 3;
        k2 = mean(S(2,:).^4) / (mean(S(2,:).^2)^2) - 3;
        
        val = abs(k1) + abs(k2);
        
        if val > best_val
            best_val = val;
            best_theta = theta;
        end
    end
    
    R_hat = [cos(best_theta), -sin(best_theta); 
             sin(best_theta),  cos(best_theta)]'; 
end