function [P] = NWcorner(a, b)

n = numel(a);
m = numel(b);

P = zeros(n, m);

% 'Northwest corner' rule
i = 1;
j = 1;
r = a(1);
c = b(1);

while i <= n && j <=m
   
    t = min(r,c);
    P(i,j) = t;
    r = r-t;
    c = c-t;
    
    if r < eps
        i = i+1;
        if i<= n
            r = a(i);
        end
    end
    
    if c < eps
       j = j+1;
       if j<= m
          c = b(j); 
       end
    end
        
end

end

