%=========================================================================%
%  Uncoupled Linear Regression via Optimal Transport & Empirical Bayes
%  INTEGRATED MASTER SCRIPT (Featuring uir_1d.m NPMLE Profiling)
%=========================================================================%
clear; clc; close all; rng(20260623);

% Silence the benign ill-conditioning warnings thrown by the interior-point
% solver in nndensity.m as it approaches the optimum (the barrier Hessian
% becomes near-singular precisely *because* the duality gap -> 0).
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');

% Standard normal PDF passed to uir_1d to bypass slow built-in normpdf checks
phi = @(z) exp(-0.5*z.^2)/sqrt(2*pi);          

% Global problem settings
true_sigma = 0.20;    % True noise SD
b0         = 0.0;     % True intercept
design     = 'grid';  % 'grid' = quantile grid of U(0,1]; 'rand' = iid U(0,1)

xgen = @(m) (strcmp(design,'grid')).*((1:m)'/m) + ...
            (strcmp(design,'rand')).*sort(rand(m,1));

%% =======================================================================%
%  PART 1: TESTING NOISE RECOVERY ACROSS MULTIPLE BETAS
% ========================================================================%
fprintf('=== PART 1: Noise Scale Recovery Across Multiple Betas ===\n');
test_betas = [0.5, 1.0, 2.0, 5.0];
n_test     = 2^11;
x_test     = xgen(n_test);

fprintf('------------------------------------------------------------\n');
fprintf(' True Beta | True Sigma | Cumulant Est. | uir_1d Profile Est\n');
fprintf('------------------------------------------------------------\n');

for b = test_betas
    y_raw      = (b * x_test + b0) + true_sigma * randn(n_test, 1);
    y_shuffled = y_raw(randperm(n_test)');
    
    % Method A: 4th Cumulant Plug-in
    sig_cum = estimate_sigma_cumulant(x_test, y_shuffled);
    
    % Method B: uir_1d.m NPMLE Profiling (using 18 grid points for speed)
    [sig_npmle, ~, ~, ~] = profile_sigma_npmle(sort(x_test), sort(y_shuffled), phi, 18);
    
    fprintf(' %9.2f | %10.2f | %13.4f | %18.4f\n', b, true_sigma, sig_cum, sig_npmle);
end
fprintf('------------------------------------------------------------\n\n');


%% =======================================================================%
%  PART 2: SINGLE-N DETAILED BREAKDOWN (n = 2048)
% ========================================================================%
beta  = 2.0;
n     = 2^11;
x     = xgen(n);
theta = beta * x + b0;
y     = theta + true_sigma * randn(n, 1);
yperm = y(randperm(n)'); 
sx    = sort(x);
sy    = sort(yperm);
truth = beta * sx + b0;

mse = @(f) mean((f - truth).^2);

% 1. Estimate Noise
sig_cum = estimate_sigma_cumulant(x, yperm);
[sig_np, fhat_np, neg_ll_curve, sig_grid] = profile_sigma_npmle(sx, sy, phi, 25);

% 2. Build Bures-Wasserstein Affine Maps
mu_x = mean(x);  var_x = var(x);
mu_y = mean(sy); var_y = var(sy);

% Oracle BW Map (Knows true sigma)
b_oracle    = sqrt(max(var_y - true_sigma^2, 0) / var_x);
T_bw_oracle = mu_y + b_oracle * (sx - mu_x);

% Cumulant BW Map
b_cum    = sqrt(max(var_y - sig_cum^2, 0) / var_x);
T_bw_cum = mu_y + b_cum * (sx - mu_x);

% NPMLE Profiled BW Map
b_np    = sqrt(max(var_y - sig_np^2, 0) / var_x);
T_bw_np = mu_y + b_np * (sx - mu_x);

% Naive Uncorrected BW Map (Assumes zero noise)
b_naive    = sqrt(var_y / var_x);
T_bw_naive = mu_y + b_naive * (sx - mu_x);

fprintf('=== PART 2: Single-n Estimator Breakdown (n = %d) ===\n', n);
fprintf('  Oracle True Sigma (%.4f)   --> BW Map MSE = %.3e\n', true_sigma, mse(T_bw_oracle));
fprintf('  Cumulant Sigma    (%.4f)   --> BW Map MSE = %.3e\n', sig_cum, mse(T_bw_cum));
fprintf('  NPMLE Prof. Sigma (%.4f)   --> BW Map MSE = %.3e\n', sig_np, mse(T_bw_np));
fprintf('  NPMLE Step-Fn Direct Fit        --> Direct MSE = %.3e\n', mse(fhat_np));
fprintf('  Naive Uncorrected (Sigma=0)     --> BW Map MSE = %.3e\n\n', mse(T_bw_naive));

% Visualization
figure('Position', [100, 100, 950, 400]);

% Left: The Transport Maps
subplot(1,2,1); hold on; box on;
plot(sx, truth, 'k-', 'LineWidth', 2);
plot(sx, sy, 'Color', [0.75 0.75 0.75]);
plot(sx, T_bw_naive, 'm:', 'LineWidth', 1.5);
plot(sx, T_bw_cum, 'b--', 'LineWidth', 1.5);
plot(sx, T_bw_np, 'r-', 'LineWidth', 1.5);
plot(sx, fhat_np, 'Color', [1 0.5 0], 'LineStyle', '-.', 'LineWidth', 1.2);
legend('True Signal (\beta X)', 'Sorted Shuffled Y', 'BW (Uncorrected)', ...
       'BW (\sigma_{Cumulant})', 'BW (\sigma_{NPMLE})', 'uir\_1d Direct Step Fit', ...
       'Location', 'northwest');
xlabel('Sorted X'); ylabel('Response'); title('Transport Maps & Denoisers');

% Right: The uir_1d Profile Objective Curve (Negative Log-Likelihood)
subplot(1,2,2); hold on; box on;
plot(sig_grid, neg_ll_curve, 'k-', 'LineWidth', 1.5);
xline(true_sigma, 'g--', 'True \sigma', 'LineWidth', 1.5);
plot(sig_np, min(neg_ll_curve), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
xlabel('Candidate \sigma_0'); ylabel('Negative Log-Likelihood (Minimize)');
title('uir\_1d Profile Likelihood Minimizer');


%% =======================================================================%
%  PART 3: ASYMPTOTIC RATE CHECK (O_P(1/n))
% ========================================================================%
fprintf('=== PART 3: Asymptotic Convergence Rate (Target slope ~= -1.0) ===\n');
ns   = 2.^(7:12);
reps = 15; % Kept at 15 so the quadratic solver loop finishes in ~45 seconds

MSE_mat = zeros(numel(ns), 3); % Columns: [Oracle, Cumulant, NPMLE]

for k = 1:numel(ns)
    nk = ns(k);
    errs = zeros(reps, 3);
    
    for r = 1:reps
        xk  = xgen(nk);
        thk = beta * xk + b0;
        yk  = thk + true_sigma * randn(nk, 1);
        ypk = yk(randperm(nk)');
        sxk = sort(xk); 
        syk = sort(ypk);
        trk = beta * sxk + b0;
        
        vx = var(xk); vy = var(ypk); my = mean(ypk); mx = mean(xk);
        
        % 1. Oracle Plug-in
        bh_or = sqrt(max(vy - true_sigma^2, 0) / vx);
        errs(r, 1) = mean(((my + bh_or*(sxk - mx)) - trk).^2);
        
        % 2. Cumulant Plug-in
        s_cum = estimate_sigma_cumulant(xk, ypk);
        bh_cum = sqrt(max(vy - s_cum^2, 0) / vx);
        errs(r, 2) = mean(((my + bh_cum*(sxk - mx)) - trk).^2);
        
        % 3. uir_1d NPMLE Plug-in (using a fast 12-point sweep for the simulation)
        [s_np, ~, ~, ~] = profile_sigma_npmle(sxk, syk, phi, 12);
        bh_np = sqrt(max(vy - s_np^2, 0) / vx);
        errs(r, 3) = mean(((my + bh_np*(sxk - mx)) - trk).^2);
    end
    MSE_mat(k, :) = mean(errs, 1);
end

% Extract log-log slopes via Polyfit
slopes = zeros(1,3);
for j = 1:3
    p = polyfit(log(ns'), log(MSE_mat(:, j)), 1);
    slopes(j) = p(1);
end

fprintf('  Log-log slope (Oracle)   : %.3f\n', slopes(1));
fprintf('  Log-log slope (Cumulant) : %.3f\n', slopes(2));
fprintf('  Log-log slope (uir_1d)   : %.3f\n\n', slopes(3));

figure('Color', 'k');

% 1. Plot the estimator MSEs on log-log scales
loglog(ns, MSE_mat(:,1), '-o', 'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', 'Oracle \sigma');
hold on;
loglog(ns, MSE_mat(:,2), '-s', 'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', '4th Cumulant');
loglog(ns, MSE_mat(:,3), '-^', 'LineWidth', 1.6, 'MarkerSize', 6, 'DisplayName', 'NPMLE Profile');

% 2. Create the "Ground Truth" parametric reference slope (-1.0)
% We anchor it to the first point of the Oracle line for a clean visual offset
ref_line = MSE_mat(1,1) * (ns(1) ./ ns);
loglog(ns, ref_line, 'y--', 'LineWidth', 1.3, 'DisplayName', 'O(1/n) Reference');

% 3. Polish the grid (Crucial for log plots so the eye can track decades)
grid on; 
grid minor;
set(gca, 'Box', 'on', 'FontSize', 11, 'TickLabelInterpreter', 'latex');

% 4. Attach formal LaTeX labels
xlabel('Sample Size ($n$)', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Mean Squared Error ($\mathcal{L}_2$ Loss)', 'Interpreter', 'latex', 'FontSize', 13);
title('Parametric Rate of Convergence ($n^{-1}$)', 'Interpreter', 'latex', 'FontSize', 14.5);

legend('Location', 'southwest', 'Interpreter', 'latex', 'FontSize', 11);


%% =======================================================================%
%  LOCAL HELPER FUNCTIONS
% ========================================================================%

function sigma_est = estimate_sigma_cumulant(x, y)
    % Resolves noise scale via excess kurtosis ratio
    var_x = var(x, 1); var_y = var(y, 1);
    k4_x  = mean((x - mean(x)).^4) - 3 * (var_x^2);
    k4_y  = mean((y - mean(y)).^4) - 3 * (var_y^2);
    
    if abs(k4_x) < 1e-12 || (sign(k4_x) ~= sign(k4_y))
        sigma_est = 0; return;
    end
    
    sig2_est  = var_y - var_x * sqrt(max(k4_y / k4_x, 0));
    sigma_est = sqrt(max(sig2_est, 0));
end


function [sig_hat, full_fhat, neg_ll_curve, sig_grid] = profile_sigma_npmle(sx, sy_sorted, phi, n_points)
    % Profile the noise level sigma by MINIMIZING THE NPMLE MARGINAL
    % NEGATIVE LOG-LIKELIHOOD returned by uir_1d (Kiefer-Wolfowitz), then
    % refine to sub-grid resolution.
    %
    % WHY THE OLD VERSION GAVE A FLAT SLOPE (~-0.05):
    %   The previous code scored each candidate sigma with a Gaussian
    %   residual-sum-of-squares likelihood built from the *fitted* monotone
    %   map:  (0.5/s0^2)*rss + n*log(s0).  But the fitted map interpolates
    %   the sorted data more and more tightly as s0 -> 0, so rss collapses
    %   while n*log(s0) -> -Inf.  The objective therefore had NO interior
    %   minimum and the minimizer was always pinned to the smallest grid
    %   point => sigma underestimated => the BW map collapsed to the naive
    %   (sigma = 0) map => a constant bias whose MSE never shrank with n.
    %
    %   uir_1d's 3rd output is the proper marginal likelihood of the
    %   deconvolution mixture.  With only ~sqrt(n) location nodes, a
    %   too-small sigma leaves gaps between nodes (huge -log penalty) and a
    %   too-large sigma over-smooths, so the objective now has a genuine
    %   interior minimum near the true noise level.

    if nargin < 4; n_points = 20; end
    n = numel(sy_sorted);

    % Grid kept safely away from 0 so phi(z/s)/s never blows up
    sig_max  = std(sy_sorted);
    sig_grid = linspace(0.02 * sig_max, 0.82 * sig_max, n_points);
    neg_ll_curve = inf(size(sig_grid));

    m_nodes = ceil(sqrt(n));           % location-grid density

    for k = 1:numel(sig_grid)
        s0 = sig_grid(k);
        % uir_1d already runs silently now (nndensity verbose flag = false),
        % so no evalc / warning('off','all') gymnastics are needed.
        [~, ~, obj] = uir_1d(sx, sy_sorted, s0, m_nodes, true, phi);
        if isfinite(obj)
            neg_ll_curve(k) = obj;     % marginal negative log-likelihood
        end
    end

    [~, best_idx] = min(neg_ll_curve);
    sig_hat = sig_grid(best_idx);

    % --- sub-grid parabolic refinement ------------------------------------
    % A fixed n_points grid quantizes sigma_hat, leaving a constant error
    % floor that would by itself flatten the convergence curve.  Fitting a
    % parabola to the three points around the grid minimizer recovers a
    % continuous estimate that can actually converge as n grows.
    if best_idx > 1 && best_idx < numel(sig_grid)
        Lm = neg_ll_curve(best_idx - 1);
        L0 = neg_ll_curve(best_idx);
        Lp = neg_ll_curve(best_idx + 1);
        denom = Lm - 2*L0 + Lp;
        if all(isfinite([Lm L0 Lp])) && denom > 0       % locally convex
            h     = sig_grid(2) - sig_grid(1);
            delta = 0.5 * h * (Lm - Lp) / denom;
            delta = max(min(delta, h), -h);             % stay in the bracket
            sig_hat = sig_grid(best_idx) + delta;
        end
    end

    % Final monotone push-forward map at the selected sigma
    [x_unq, fhat_unq] = uir_1d(sx, sy_sorted, sig_hat, m_nodes, true, phi);
    if numel(fhat_unq) == n
        full_fhat = fhat_unq;
    else
        full_fhat = interp1(x_unq, fhat_unq, sx, 'previous', 'extrap');
    end
end