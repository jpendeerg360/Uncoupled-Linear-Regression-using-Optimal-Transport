%=========================================================================%
% Uncoupled Linear Regression via Optimal Transport
% Confidence Regions (Known Noise Case) - CLT vs Bootstrap
%=========================================================================%
clear; clc; close all; rng(20260716);

% =========================================================================
% SIMULATION PARAMETERS
% =========================================================================
n_mc       = 1000;    % Monte Carlo iterations for coverage test
n_samples  = 1000;    % Sample size per iteration
n_boot     = 500;     % Bootstrap resamples

beta_true  = 2.0;     % True slope
b0_true    = 0.5;     % True intercept
sigma_true = 0.5;     % KNOWN noise scale
alpha      = 0.05;    % Significance level (95% Confidence)

chi2_crit = chi2inv(1 - alpha, 2); 
z_crit    = norminv(1 - alpha/2);  

xgen = @(m) 2 * rand(m, 1) + 1; % Uniform(1, 3)

fprintf('=== Testing 95%% Confidence Regions (Known Noise Case) ===\n');
fprintf('Running %d Monte Carlo simulations (n = %d)...\n', n_mc, n_samples);

clt_coverage  = 0; 
boot_coverage = 0;

wb = waitbar(0, 'Running Monte Carlo Coverage Tests...');

for mc = 1:n_mc
    % 1. CORRECTED DATA GENERATION: Fully Independent Draws
    X_observed = xgen(n_samples);
    X_latent   = xgen(n_samples); % Independent draw for the Y generation
    
    Y_clean = beta_true * X_latent + b0_true;
    Y_observed = Y_clean + sigma_true * randn(n_samples, 1);
    
    % We assign these to X and Y for clarity in the formulas below
    X = X_observed;
    Y = Y_observed; 
    
    % 2. Marginal Sample Moments
    mx = mean(X); vx = var(X, 1); 
    my = mean(Y); vy = var(Y, 1);
    
    m3x = mean((X - mx).^3); m4x = mean((X - mx).^4);
    m3y = mean((Y - my).^3); m4y = mean((Y - my).^4);
    
    % 3. Point Estimates
    b_hat  = sqrt(max(vy - sigma_true^2, 0) / vx);
    b0_hat = my - b_hat * mx;
    theta_hat = [b0_hat; b_hat];
    theta_true = [b0_true; beta_true];
    
    % =====================================================================
    % METHOD A: CLT (Asymptotic Delta Method)
    % =====================================================================
    V = [vx,   0,   m3x,           0;
         0,    vy,  0,             m3y;
         m3x,  0,   m4x - vx^2,    0;
         0,    m3y, 0,             m4y - vy^2];
     
    J = [-b_hat, 1, (b_hat * mx)/(2 * vx), -mx/(2 * b_hat * vx);
          0,     0, -b_hat/(2 * vx),        1/(2 * b_hat * vx)];
      
    Sigma_theta_clt = (1/n_samples) * (J * V * J');
    
    diff = theta_hat - theta_true;
    mahalanobis_clt = diff' * (Sigma_theta_clt \ diff);
    if mahalanobis_clt <= chi2_crit
        clt_coverage = clt_coverage + 1;
    end
    
    % =====================================================================
    % METHOD B: Non-parametric Bootstrap
    % =====================================================================
    boot_thetas = zeros(2, n_boot);
    for b = 1:n_boot
        X_star = X(randi(n_samples, n_samples, 1));
        Y_star = Y(randi(n_samples, n_samples, 1));
        
        vx_star = var(X_star, 1); vy_star = var(Y_star, 1);
        mx_star = mean(X_star);   my_star = mean(Y_star);
        
        b_boot  = sqrt(max(vy_star - sigma_true^2, 0) / vx_star);
        b0_boot = my_star - b_boot * mx_star;
        boot_thetas(:, b) = [b0_boot; b_boot];
    end
    
    Sigma_theta_boot = cov(boot_thetas');
    mahalanobis_boot = diff' * (Sigma_theta_boot \ diff);
    if mahalanobis_boot <= chi2_crit
        boot_coverage = boot_coverage + 1;
    end
    
    waitbar(mc/n_mc, wb);
end
close(wb);

cov_clt_pct = (clt_coverage / n_mc) * 100;
cov_boot_pct = (boot_coverage / n_mc) * 100;
fprintf('CLT Wald Region Coverage:       %.2f%%\n', cov_clt_pct);
fprintf('Bootstrap Wald Region Coverage: %.2f%%\n', cov_boot_pct);

% =========================================================================
% VISUALIZATION: Confidence Bands for a Single Realization
% =========================================================================
sx = sort(X);
X_mat = [ones(n_samples, 1), sx]; 

% CLT Pointwise Bands
SE_clt = sqrt(sum((X_mat * Sigma_theta_clt) .* X_mat, 2)); 
T_hat = X_mat * theta_hat;
clt_upper = T_hat + z_crit * SE_clt;
clt_lower = T_hat - z_crit * SE_clt;

% Bootstrap Pointwise Bands
T_boot_evals = X_mat * boot_thetas; 
boot_upper = prctile(T_boot_evals, 100 - (alpha/2)*100, 2);
boot_lower = prctile(T_boot_evals, (alpha/2)*100, 2);

% --- PLOTTING (Tabbed UI) ---
masterFig = figure('Name', 'Confidence Regions Testing', ...
                   'Position', [150, 150, 1050, 500], 'Color', 'w');
tgroup = uitabgroup(masterFig);

% TAB 1: Coverage Bar Chart
tab1 = uitab(tgroup, 'Title', 'Empirical Coverage');
ax1 = axes('Parent', tab1); hold(ax1, 'on'); box(ax1, 'on');

bars = bar(ax1, [cov_clt_pct, cov_boot_pct], 'FaceColor', 'flat');
bars.CData(1,:) = [0.2 0.6 0.8]; 
bars.CData(2,:) = [0.8 0.3 0.3]; 
yline(ax1, 95, 'g--', 'Nominal 95% Target', 'LineWidth', 2, 'LabelHorizontalAlignment', 'center', 'Interpreter', 'latex', 'FontSize', 12);

set(ax1, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'XTick', 1:2, ...
    'XTickLabel', {'CLT (Delta Method)', 'Non-Parametric Bootstrap'}, ...
    'TickLabelInterpreter', 'latex', 'FontSize', 12, 'YLim', [85 100]);
ylabel(ax1, 'Empirical Coverage (\%)', 'Interpreter', 'latex', 'FontSize', 14);
title(ax1, sprintf('Joint Parameter Coverage (N = %d, Iterations = %d)', n_samples, n_mc), 'Interpreter', 'latex', 'FontSize', 15);


% TAB 2: Split Pointwise Confidence Bands
tab2 = uitab(tgroup, 'Title', 'Pointwise Confidence Bands');
tl2 = tiledlayout(tab2, 1, 2, 'TileSpacing', 'compact');
fill_band = @(ax, x, upper, lower, color) fill(ax, [x; flipud(x)], [upper; flipud(lower)], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Subplot 1: CLT Band
ax2a = nexttile(tl2); hold(ax2a, 'on'); box(ax2a, 'on');
h_clt = fill_band(ax2a, sx, clt_upper, clt_lower, 'b');
plot(ax2a, sx, beta_true * sx + b0_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Signal');
plot(ax2a, sx, T_hat, 'k--', 'LineWidth', 1.5, 'DisplayName', '$\hat{T}_{BW}$ Estimate');

set(ax2a, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'TickLabelInterpreter', 'latex', 'FontSize', 11);
xlabel(ax2a, 'Covariate Space ($X$)', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax2a, 'Transported Target ($Y$)', 'Interpreter', 'latex', 'FontSize', 13);
title(ax2a, 'CLT 95\% Pointwise Band', 'Interpreter', 'latex', 'FontSize', 14);
lgd_a = legend(ax2a, [h_clt, findobj(ax2a, 'DisplayName', 'True Signal'), findobj(ax2a, 'DisplayName', '$\hat{T}_{BW}$ Estimate')], ...
    {'CLT 95\% Band', 'True Signal', '$\hat{T}_{BW}$ Estimate'}, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 11);
set(lgd_a, 'TextColor', 'k', 'Color', 'w');

% Subplot 2: Bootstrap Band
ax2b = nexttile(tl2); hold(ax2b, 'on'); box(ax2b, 'on');
h_boot = fill_band(ax2b, sx, boot_upper, boot_lower, 'r');
plot(ax2b, sx, beta_true * sx + b0_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Signal');
plot(ax2b, sx, T_hat, 'k--', 'LineWidth', 1.5, 'DisplayName', '$\hat{T}_{BW}$ Estimate');

set(ax2b, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'TickLabelInterpreter', 'latex', 'FontSize', 11);
xlabel(ax2b, 'Covariate Space ($X$)', 'Interpreter', 'latex', 'FontSize', 13);
title(ax2b, 'Bootstrap 95\% Pointwise Band', 'Interpreter', 'latex', 'FontSize', 14);
lgd_b = legend(ax2b, [h_boot, findobj(ax2b, 'DisplayName', 'True Signal'), findobj(ax2b, 'DisplayName', '$\hat{T}_{BW}$ Estimate')], ...
    {'Bootstrap 95\% Band', 'True Signal', '$\hat{T}_{BW}$ Estimate'}, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 11);
set(lgd_b, 'TextColor', 'k', 'Color', 'w');