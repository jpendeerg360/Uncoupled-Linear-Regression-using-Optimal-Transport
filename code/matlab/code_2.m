%=========================================================================%
% Uncoupled Linear Regression via Optimal Transport
% Confidence Regions (Estimated Noise Case via 4th Cumulant)
% Demonstrating the necessity of the Updated Bootstrap (Sec 5.2.2)
%=========================================================================%
clear; clc; close all; rng(20260716);

% =========================================================================
% SIMULATION PARAMETERS
% =========================================================================
n_mc       = 1000;    % Monte Carlo iterations
n_samples  = 1024;    % Sample size (needs to be larger for stable 4th cumulants)
n_boot     = 400;     % Bootstrap resamples

beta_true  = 2.0;     % True slope
b0_true    = 0.5;     % True intercept
sigma_true = 0.5;     % True noise scale (Unknown to the estimator)
alpha      = 0.05;    % Significance level (95% Confidence)

chi2_crit = chi2inv(1 - alpha, 2); 
z_crit    = norminv(1 - alpha/2);  

% Design generation: Uniform(1, 3) provides negative excess kurtosis
% This ensures k4(X) is non-zero, making sigma identifiable.
xgen = @(m) 2 * rand(m, 1) + 1; 

fprintf('=== Testing 95%% Confidence Regions (Estimated Noise Case) ===\n');
fprintf('Comparing Updated Bootstrap (Sec 5.2.2) vs Naive Bootstrap...\n');

updated_coverage = 0; 
naive_coverage   = 0;

wb = waitbar(0, 'Running Monte Carlo Coverage Tests...');

for mc = 1:n_mc
    % 1. DATA GENERATION: Fully Independent Draws
    X_observed = xgen(n_samples);
    X_latent   = xgen(n_samples); 
    
    Y_clean = beta_true * X_latent + b0_true;
    Y_observed = Y_clean + sigma_true * randn(n_samples, 1);
    
    X = X_observed;
    Y = Y_observed; 
    
    % 2. Parent Sample Point Estimates (4th Cumulant Method)
    mx = mean(X); vx = var(X, 1);
    my = mean(Y); vy = var(Y, 1);
    
    k4x = mean((X - mx).^4) - 3 * vx^2;
    k4y = mean((Y - my).^4) - 3 * vy^2;
    
    % Parent estimated sigma squared
    if abs(k4x) > 1e-8 && sign(k4x) == sign(k4y)
        sig2_hat = vy - vx * sqrt(k4y / k4x);
    else
        sig2_hat = 0; % Fallback if cumulants fail
    end
    sig2_hat = max(sig2_hat, 0);
    
    b_hat  = sqrt(max(vy - sig2_hat, 0) / vx);
    b0_hat = my - b_hat * mx;
    
    theta_hat  = [b0_hat; b_hat];
    theta_true = [b0_true; beta_true];
    
    % =====================================================================
    % BOOTSTRAP LOOP
    % =====================================================================
    boot_thetas_updated = zeros(2, n_boot);
    boot_thetas_naive   = zeros(2, n_boot);
    
    for b = 1:n_boot
        X_star = X(randi(n_samples, n_samples, 1));
        Y_star = Y(randi(n_samples, n_samples, 1));
        
        vx_star = var(X_star, 1); vy_star = var(Y_star, 1);
        mx_star = mean(X_star);   my_star = mean(Y_star);
        
        % --- NAIVE BOOTSTRAP ---
        % Treats the parent sig2_hat as an absolute known constant.
        b_naive  = sqrt(max(vy_star - sig2_hat, 0) / vx_star);
        b0_naive = my_star - b_naive * mx_star;
        boot_thetas_naive(:, b) = [b0_naive; b_naive];
        
        % --- UPDATED BOOTSTRAP (Section 5.2.2) ---
        % Re-estimates noise scale from the resampled 4th cumulants.
        k4x_star = mean((X_star - mx_star).^4) - 3 * vx_star^2;
        k4y_star = mean((Y_star - my_star).^4) - 3 * vy_star^2;
        
        if abs(k4x_star) > 1e-8 && sign(k4x_star) == sign(k4y_star)
            sig2_star = vy_star - vx_star * sqrt(k4y_star / k4x_star);
        else
            sig2_star = 0;
        end
        sig2_star = max(sig2_star, 0);
        
        b_up  = sqrt(max(vy_star - sig2_star, 0) / vx_star);
        b0_up = my_star - b_up * mx_star;
        boot_thetas_updated(:, b) = [b0_up; b_up];
    end
    
    % Wald Tests for Coverage
    diff = theta_hat - theta_true;
    
    Sigma_up = cov(boot_thetas_updated');
    if diff' * (Sigma_up \ diff) <= chi2_crit
        updated_coverage = updated_coverage + 1;
    end
    
    Sigma_naive = cov(boot_thetas_naive');
    if diff' * (Sigma_naive \ diff) <= chi2_crit
        naive_coverage = naive_coverage + 1;
    end
    
    waitbar(mc/n_mc, wb);
end
close(wb);

cov_up_pct = (updated_coverage / n_mc) * 100;
cov_naive_pct = (naive_coverage / n_mc) * 100;
fprintf('Updated Bootstrap (Sec 5.2.2) Coverage: %.2f%%\n', cov_up_pct);
fprintf('Naive Bootstrap (Fixed Sigma) Coverage: %.2f%%\n', cov_naive_pct);

% =========================================================================
% VISUALIZATION
% =========================================================================
sx = sort(X);
X_mat = [ones(n_samples, 1), sx]; 

% Pointwise Bands (Using the last MC iteration's bootstraps)
T_up_evals = X_mat * boot_thetas_updated; 
up_upper = prctile(T_up_evals, 100 - (alpha/2)*100, 2);
up_lower = prctile(T_up_evals, (alpha/2)*100, 2);

T_naive_evals = X_mat * boot_thetas_naive; 
naive_upper = prctile(T_naive_evals, 100 - (alpha/2)*100, 2);
naive_lower = prctile(T_naive_evals, (alpha/2)*100, 2);

T_hat = X_mat * theta_hat;

% --- PLOTTING (Tabbed UI) ---
masterFig = figure('Name', 'Estimated Noise Confidence Regions', ...
                   'Position', [150, 150, 1050, 500], 'Color', 'w');
tgroup = uitabgroup(masterFig);

% TAB 1: Coverage Bar Chart
tab1 = uitab(tgroup, 'Title', 'Empirical Coverage');
ax1 = axes('Parent', tab1); hold(ax1, 'on'); box(ax1, 'on');

bars = bar(ax1, [cov_up_pct, cov_naive_pct], 'FaceColor', 'flat');
bars.CData(1,:) = [0.2 0.8 0.4]; % Green for correct Updated
bars.CData(2,:) = [0.8 0.3 0.3]; % Red for failing Naive
yline(ax1, 95, 'g--', 'Nominal 95% Target', 'LineWidth', 2, 'LabelHorizontalAlignment', 'center', 'Interpreter', 'latex', 'FontSize', 12);

set(ax1, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'XTick', 1:2, ...
    'XTickLabel', {'Updated Bootstrap (Sec 5.2.2)', 'Naive Bootstrap (Fixed $\hat{\sigma}$)'}, ...
    'TickLabelInterpreter', 'latex', 'FontSize', 12, 'YLim', [40 100]);
ylabel(ax1, 'Empirical Coverage (\%)', 'Interpreter', 'latex', 'FontSize', 14);
title(ax1, sprintf('Joint Parameter Coverage (Estimated Noise, $N = %d$)', n_samples), 'Interpreter', 'latex', 'FontSize', 15);

% TAB 2: Split Pointwise Confidence Bands
tab2 = uitab(tgroup, 'Title', 'Pointwise Confidence Bands');
tl2 = tiledlayout(tab2, 1, 2, 'TileSpacing', 'compact');
fill_band = @(ax, x, upper, lower, color) fill(ax, [x; flipud(x)], [upper; flipud(lower)], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Subplot 1: Updated Bootstrap Band
ax2a = nexttile(tl2); hold(ax2a, 'on'); box(ax2a, 'on');
h_up = fill_band(ax2a, sx, up_upper, up_lower, 'g');
plot(ax2a, sx, beta_true * sx + b0_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Signal');
plot(ax2a, sx, T_hat, 'k--', 'LineWidth', 1.5, 'DisplayName', '$\hat{T}_{BW}$ Estimate');

set(ax2a, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'TickLabelInterpreter', 'latex', 'FontSize', 11);
xlabel(ax2a, 'Covariate Space ($X$)', 'Interpreter', 'latex', 'FontSize', 13);
ylabel(ax2a, 'Transported Target ($Y$)', 'Interpreter', 'latex', 'FontSize', 13);
title(ax2a, 'Updated Bootstrap 95\% Band', 'Interpreter', 'latex', 'FontSize', 14);
lgd_a = legend(ax2a, [h_up, findobj(ax2a, 'DisplayName', 'True Signal'), findobj(ax2a, 'DisplayName', '$\hat{T}_{BW}$ Estimate')], ...
    {'Updated 95\% Band', 'True Signal', '$\hat{T}_{BW}$ Estimate'}, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 11);
set(lgd_a, 'TextColor', 'k', 'Color', 'w');

% Subplot 2: Naive Bootstrap Band
ax2b = nexttile(tl2); hold(ax2b, 'on'); box(ax2b, 'on');
h_naive = fill_band(ax2b, sx, naive_upper, naive_lower, 'r');
plot(ax2b, sx, beta_true * sx + b0_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Signal');
plot(ax2b, sx, T_hat, 'k--', 'LineWidth', 1.5, 'DisplayName', '$\hat{T}_{BW}$ Estimate');

set(ax2b, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'TickLabelInterpreter', 'latex', 'FontSize', 11);
xlabel(ax2b, 'Covariate Space ($X$)', 'Interpreter', 'latex', 'FontSize', 13);
title(ax2b, 'Naive Bootstrap 95\% Band', 'Interpreter', 'latex', 'FontSize', 14);
lgd_b = legend(ax2b, [h_naive, findobj(ax2b, 'DisplayName', 'True Signal'), findobj(ax2b, 'DisplayName', '$\hat{T}_{BW}$ Estimate')], ...
    {'Naive 95\% Band', 'True Signal', '$\hat{T}_{BW}$ Estimate'}, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 11);
set(lgd_b, 'TextColor', 'k', 'Color', 'w');