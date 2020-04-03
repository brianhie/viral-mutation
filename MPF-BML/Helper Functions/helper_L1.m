function [w] = helper_L1(funObj,w,ind_diag,ind_nodiag,options_MPF )

verbose = options_MPF.verbose;
opt_tol = options_MPF.opt_tol;
prog_tol = options_MPF.prog_tol;
max_iter = options_MPF.max_iter;
suffDec = options_MPF.suffDec;
memory = options_MPF.memory;
lambda_h = options_MPF.lambda_h;
lambda_J = options_MPF.lambda_J;
gamma_h = options_MPF.gamma_h;
gamma_J = options_MPF.gamma_J;


if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
w = [w.*(w>0);-w.*(w<0)];

[f,g] =nonNegGrad(funObj,w,p,lambda_J,lambda_h,gamma_J,gamma_h,ind_diag,ind_nodiag );
funEvals = 1;

% Compute working set and check optimality
W = (w~=0) | (g < 0);
optCond = max(abs(g(W)));
if optCond < opt_tol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    w = w(1:p)-w(p+1:end);
    return;
end

%% Main loop
for i = 1:max_iter
    
    % Compute direction
    if i == 1
        d = -g;
        t = min(1,1/sum(abs(g)));
        old_fvals = repmat(-inf,[memory 1]);
        fr = f;
    else
        y = g-g_old;
        s = w-w_old;

        alpha = (y'*s)/(y'*y);
%         if alpha <= 1e-10 || alpha > 1e10
%             fprintf('BB update is having some trouble, implement fix!\n');
% %             pause;
%         end
        d = -alpha*g;
        t = 1;

        if i-1 <= memory
            old_fvals(i-1) = f;
        else
            old_fvals = [old_fvals(2:end);f];
        end
        fr = max(old_fvals);
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    % Compute directional derivative, check that we can make progress
    gtd = g'*d;
    if gtd > -prog_tol
        if verbose
            fprintf('Directional derivative below prog_tol\n');
        end
        break;
    end
    
    % Compute projected point
    w_new = nonNegProject(w+t*d);
    [f_new,g_new] =nonNegGrad(funObj,w_new,p,lambda_J,lambda_h,gamma_J,gamma_h,ind_diag,ind_nodiag );

    funEvals = funEvals+1;
    
    backtrackind=0;
    % Line search along projection arc
    while f_new > fr + suffDec*g'*(w_new-w) || ~isLegal(f_new)
        t_old = t;
        
        % Backtracking
        if verbose
            fprintf('Backtracking...\n');
        end
        if ~isLegal(f_new)
            if verbose
                fprintf('Halving Step Size\n');
            end
            t = .5*t;
        else
            t = polyinterp([0 f gtd; t f_new g_new'*d]);
        end
        
        % Adjust if interpolated value near boundary
        if t < t_old*1e-3
            if verbose == 3
                fprintf('Interpolated value too small, Adjusting\n');
            end
            t = t_old*1e-3;
        elseif t > t_old*0.6
            if verbose == 3
                fprintf('Interpolated value too large, Adjusting\n');
            end
            t = t_old*0.6;
        end
        
        % Check whether step has become too small
        if max(abs(t*d)) < prog_tol
            if verbose
                fprintf('Step too small in line search\n');
            end
            t = 0;
            w_new = w;
            f_new = f;
            g_new = g;
            break;
        end
        
        % Compute projected point
        w_new = nonNegProject(w+t*d);
            [f_new,g_new] =nonNegGrad(funObj,w_new,p,lambda_J,lambda_h,gamma_J,gamma_h,ind_diag,ind_nodiag );

        funEvals = funEvals+1;
        
        backtrackind=backtrackind+1;
        
        if backtrackind==5
            break;
        end
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    
    % Compute new working set
    W = (w~=0) | (g < 0);
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,t,f,max(abs(g(W))),nnz(w(1:p)-w(p+1:end)));
    end
    
    % Check Optimality
    optCond = max(abs(g(W)));
    if optCond < opt_tol
        if verbose
            fprintf('First-order optimality below opt_tol\n');
        end
        break;
    end
    
    % Check for lack of progress
    if max(abs(t*d)) < prog_tol || abs(f-f_old) < prog_tol
        if verbose
        fprintf('Progress in parameters or objective below prog_tol\n');
        end
        break;
    end
    
    % Check for iteration limit
    if funEvals >= max_iter
        if verbose
            fprintf('Function evaluations reached max_iter\n');
        end
        break;
    end
end
% w=w';
w = w(1:p)-w(p+1:end);

end

%% Non-negative variable gradient calculation
function [f,g,H] = nonNegGrad(funObj,w,p,lambda_J,lambda_h,gamma_J,gamma_h,ind_diag,ind_nodiag )


[f,g] = funObj(w(1:p)-w(p+1:end) );
%%%%%%%%%%%%%%%%%%%
Jflat = w(1:p)-w(p+1:end) ;
ndims=sqrt(p);
J = reshape(Jflat,ndims,ndims);
J = (J + J')/2;

% L2 regularization
Jnew=J- diag(diag(J));
reg_mat1 = gamma_J*sum(sum((Jnew.^2)))/2;
reg_diff_mat1  = 2*gamma_J*(reshape(Jnew,1,ndims*ndims)')/2;

Jnew=diag(diag(J));
reg_mat2 = gamma_h*sum(sum((Jnew.^2)))/2;
reg_diff_mat2  = 2*gamma_h*(reshape(Jnew,1,ndims*ndims)')/2;

reg_mat=reg_mat1+reg_mat2;
reg_diff_mat=reg_diff_mat1+reg_diff_mat2;

f = f + reg_mat;
g  = g + reg_diff_mat;

% L1 regularization
w1 = w(1:p);
w2 = w(p+1:end);
f = f + lambda_h*sum(w1(ind_diag)) + lambda_h*sum(w2(ind_diag))+ sum(lambda_J.*w2(ind_nodiag)) + lambda_J*sum(w2(ind_nodiag));

g1 = ones(p,1);
g1(ind_diag) = lambda_h.*ones(length(ind_diag),1);
g1(ind_nodiag) = lambda_J.*ones(length(ind_nodiag),1);
g = [g;-g] + [g1;g1];


end

function [w] = nonNegProject(w)
w(w < 0) = 0;
end
