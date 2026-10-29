% * Monotone Unlinked Regression in 1d *
% -based on Kiefer-Wolfowitz NPMLE- 
%
% x: design points (ordered) --- ties allowed
% y: responses (ordered)
% sigma: noise level
% m: number of mu's (default: sqrt(#design points)
% nw: true/false --- apply the northwest corner rule (OT)
% pdf: pdf of the noises (default: standard normal PDF), function handle

function [xunique, fhat, obj] = uir_1d(x, y, sigma, m, nw, pdf)

n = numel(y);

if nargin < 6
   pdf = @normpdf; 
end

if nargin < 5
   nw = true; 
end

if nargin < 4
   m = ceil(sqrt(n)); 
end

% define grid for location parameters
if ~isfinite(m)
   mu = sort(y)';
   m =n;
else
    mu = linspace(min(y), max(y), m);
end

Phi = zeros(n, m);

% create matrix of PDF evaluations
for j=1:m
    Phi(:, j) = pdf((y - mu(j))/sigma)/sigma;  
end

nullind = sum(Phi) < 1E-8;
Phi = Phi(:,~nullind);
[alpha_tmp, ~, ~, obj] = nndensity(Phi, ones(size(Phi,2),1)/size(Phi,2), false);
alpha = zeros(numel(nullind), 1);
alpha(~nullind) = alpha_tmp;


% C = -2*sqrt(x)*mu + sqrt(x).^2 * ones(1,m) + ones(n,1) * ((mu').^2)';
% 
% cvx_begin
% cvx_quiet(true)
% variable P(n, m)
% dual variable f
% dual variable g
% minimize sum(C(:) .* P(:))
% subject to
% f: sum(P, 2) == (ones(size(y, 1), 1)/size(y, 1))
% g: sum(P,1) == alpha'
% P >= 0
% cvx_end

% Pback = P;

T = tabulate(x);
xunique = T(:,1);
a = T(:,2)/n;

if nw
    P = NWcorner(a, alpha);
    fhat = (P*mu') ./ a;
else
    fhat = zeros(numel(xunique), 1);
    cs = cumsum(alpha);
    cs_a = cumsum(a);
    for i=1:numel(xunique)    
        fhat(i) = mu(find(cs + sqrt(eps) >= cs_a(i), 1, 'first'));
    end
end


    
end

