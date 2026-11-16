clear; clc; close all;




%% ============================================================
% USER SETTINGS
%% ============================================================

coarse_method = 'IM2';      % 'BE', 'IM2', 'RK3', 'RK4'
fine_method   = 'RK4';      % 'BE', 'IM2', 'RK3', 'RK4'


%% ============================================================
% Problem parameters
%% ============================================================

d     = 0.01;
alpha = 1.0;
T     = 1.0;


%% ============================================================
% Fixed spatial mesh
%% ============================================================

J = 50;
h = 1/J;

x = (1:J-1)'*h;
m = J-1;


%% ============================================================
% Spatial operator
%% ============================================================

e = ones(m,1);

A_h = (d/h^2) * ...
      spdiags([e,-2*e,e],-1:1,m,m);

L_h = A_h + alpha*speye(m);


%% ============================================================
% Initial condition
%% ============================================================

v1 = sin(pi*x);
v2 = sin(2*pi*x);

U0 = v1 + 0.5*v2;


%% ============================================================
% Exact semidiscrete solution
%% ============================================================

mu1_h = 4/h^2*sin(pi*h/2)^2;
mu2_h = 4/h^2*sin(pi*h)^2;

lambda1_h = alpha-d*mu1_h;
lambda2_h = alpha-d*mu2_h;

U_exact_h = @(t) ...
      exp(lambda1_h*t)*v1 ...
    + 0.5*exp(lambda2_h*t)*v2;


%% ============================================================
% Parareal parameters
%% ============================================================

N_list = [20,40,80,160];

M = 5;
K = 3;

numN = length(N_list);

Error = zeros(numN,K+1);
Order = NaN(numN,K+1);
Tmax  = zeros(numN,K+1);


%% ============================================================
% Method orders
%% ============================================================

p = method_order(coarse_method);
q = method_order(fine_method);

fprintf('\n');
fprintf('=============================================\n');

fprintf('Coarse propagator : %s, order p = %d\n', ...
    method_full_name(coarse_method),p);

fprintf('Fine propagator   : %s, order q = %d\n', ...
    method_full_name(fine_method),q);

fprintf('=============================================\n');

if p > q

    warning(['The convergence theory assumes p <= q, ' ...
             'but the current choice gives p > q.']);

end


%% ============================================================
% Main loop
%% ============================================================

for iN = 1:numN

    N = N_list(iN);

    DeltaT = T/N;
    dt     = DeltaT/M;

    fprintf('\n');

    fprintf('Computing N = %d, DeltaT = %.6e, dt = %.6e\n', ...
        N,DeltaT,dt);


    %% --------------------------------------------------------
    % Coarse propagator
    %% --------------------------------------------------------

    G = build_stepper( ...
        L_h,DeltaT,coarse_method);


    %% --------------------------------------------------------
    % Fine propagator
    %% --------------------------------------------------------

    Fstep = build_stepper( ...
        L_h,dt,fine_method);


    %% --------------------------------------------------------
    % Storage
    %% --------------------------------------------------------

    U = zeros(m,N+1,K+1);


    %% ========================================================
    % k = 0: coarse prediction
    %% ========================================================

    U(:,1,1) = U0;

    for n = 1:N

        U(:,n+1,1) = ...
            G(U(:,n,1));

    end


    %% ========================================================
    % Parareal iterations
    %% ========================================================

    for k = 1:K

        U(:,1,k+1) = U0;

        for n = 1:N


            %% New coarse propagation

            G_new = ...
                G(U(:,n,k+1));


            %% Fine propagation

            F_old = fine_propagation( ...
                U(:,n,k), ...
                Fstep, ...
                M);


            %% Old coarse propagation

            G_old = ...
                G(U(:,n,k));


            %% Parareal correction

            U(:,n+1,k+1) = ...
                  G_new ...
                + F_old ...
                - G_old;

        end

    end


    %% ========================================================
    % Maximum-in-time errors
    %% ========================================================

    for k = 0:K

        max_err  = 0.0;
        max_time = 0.0;

        for n = 0:N

            tn = n*DeltaT;


            %% Exact semidiscrete solution

            Uh_ex = U_exact_h(tn);


            %% Error

            err_vec = ...
                U(:,n+1,k+1)-Uh_ex;


            %% Discrete L2 norm

            err_n = ...
                sqrt(h*sum(err_vec.^2));


            %% Maximum-in-time error

            if err_n > max_err

                max_err  = err_n;
                max_time = tn;

            end

        end

        Error(iN,k+1) = max_err;
        Tmax(iN,k+1)  = max_time;

    end

end


%% ============================================================
% Observed convergence orders
%
% order = log_2(E_N/E_2N)
%% ============================================================

for iN = 2:numN

    for k = 0:K

        Order(iN,k+1) = ...
            log(Error(iN-1,k+1) ...
            /Error(iN,k+1))/log(2);

    end

end


%% ============================================================
% Theoretical convergence orders
%
% r_k = min{p(k+1),q}
%% ============================================================

PredictedOrder = zeros(1,K+1);

for k = 0:K

    PredictedOrder(k+1) = ...
        min(p*(k+1),q);

end


%% ============================================================
% Display numerical results
%% ============================================================

fprintf('\n');

fprintf('=================================================================================================================\n');

fprintf(' Parareal temporal convergence test\n');

fprintf(' Coarse = %s (p=%d)\n', ...
    method_full_name(coarse_method),p);

fprintf(' Fine   = %s (q=%d)\n', ...
    method_full_name(fine_method),q);

fprintf(' M = %d, J = %d\n',M,J);

fprintf('=================================================================================================================\n');


fprintf('%6s','N');

for k = 0:K

    fprintf(' |      E(k=%d)       Order ',k);

end

fprintf('\n');

fprintf('-----------------------------------------------------------------------------------------------------------------\n');


for iN = 1:numN

    fprintf('%6d',N_list(iN));

    for k = 0:K

        if iN == 1

            fprintf(' | %13.6e   %6s', ...
                Error(iN,k+1),'--');

        else

            fprintf(' | %13.6e   %6.4f', ...
                Error(iN,k+1), ...
                Order(iN,k+1));

        end

    end

    fprintf('\n');

end

fprintf('=================================================================================================================\n');


%% ============================================================
% Predicted orders
%% ============================================================

fprintf('\nPredicted temporal orders:\n');

for k = 0:K

    fprintf('k = %d : min{%d*(%d+1), %d} = %d\n', ...
        k,p,k,q,PredictedOrder(k+1));

end


%% ============================================================
% Times of maximum errors
%% ============================================================

fprintf('\nTimes at which maximum errors occur:\n');

for iN = 1:numN

    fprintf('N = %3d : ',N_list(iN));

    for k = 0:K

        fprintf('k=%d: %.6f   ', ...
            k,Tmax(iN,k+1));

    end

    fprintf('\n');

end


%% ============================================================
% CONVERGENCE PLOT
%
% Reference-paper style:
%
%       X = log_2(N)
%       Y = log_10(Error)
%
% Hence the convergence curves go downward as N increases.
%% ============================================================


%% ------------------------------------------------------------
% Horizontal coordinate
%% ------------------------------------------------------------

X_plot = log2(N_list);


%% ------------------------------------------------------------
% Vertical shifts for clarity
%
% IMPORTANT:
% These shifts are ONLY used in the figure.
%
% Since we plot log10(Error), adding a constant produces
% a vertical shift without changing the slope.
%
% k=0 is not shifted.
%% ------------------------------------------------------------

% vertical_shift = [0, -0.25, -0.50, -0.75];
vertical_shift = [0, -0.25,1.5, -2];

%% ------------------------------------------------------------
% Logarithmic errors
%% ------------------------------------------------------------

Y_plot = zeros(size(Error));

for k = 0:K

    % Y_plot(:,k+1) = ...
    %     log10(Error(:,k+1)) ...
    %     + vertical_shift(k+1);
  Y_plot(:,k+1) = ...
         log2(Error(:,k+1)) ...
        + vertical_shift(k+1);
end


%% ============================================================
% Create figure
%% ============================================================

figure( ...
    'Color','w', ...
    'Position',[200,150,700,520]);

hold on;
box on;
grid on;


%% ============================================================
% Numerical curves
%% ============================================================

plot( ...
    X_plot, ...
    Y_plot(:,1), ...
    '-o', ...
    'LineWidth',1.2, ...
    'MarkerSize',6, ...
    'DisplayName','{\it k} = 0');


plot( ...
    X_plot, ...
    Y_plot(:,2), ...
    '-s', ...
    'LineWidth',1.2, ...
    'MarkerSize',6, ...
    'DisplayName','{\it k} = 1');


plot( ...
    X_plot, ...
    Y_plot(:,3), ...
    '-^', ...
    'LineWidth',1.2, ...
    'MarkerSize',6, ...
    'DisplayName','{\it k} = 2');


plot( ...
    X_plot, ...
    Y_plot(:,4), ...
    '-d', ...
    'LineWidth',1.2, ...
    'MarkerSize',6, ...
    'DisplayName','{\it k} = 3');


%% ============================================================
% Theoretical order reference lines
%% ============================================================

unique_orders = ...
    unique(PredictedOrder,'stable');


%% ------------------------------------------------------------
% IMPORTANT:
%
% X = log2(N)
% Y = log10(E)
%
% If E ~ N^(-r), then
%
% log10(E)
%   = C - r*log10(N)
%
% Since X = log2(N),
%
% log10(N) = log10(2)*X.
%
% Therefore the slope in this plot is
%
%       -r*log10(2)
%
% rather than simply -r.
%% ------------------------------------------------------------


%% ------------------------------------------------------------
% Additional vertical offsets of reference lines
%% ------------------------------------------------------------

% reference_offset = ...
%     [-0.35,-0.45,-0.55,-0.65];
reference_offset = ...
    [-0.35,0.8,-0.55,-0.65];

for ir = 1:length(unique_orders)

    r = unique_orders(ir);


    %% --------------------------------------------------------
    % First numerical curve with theoretical order r
    %% --------------------------------------------------------

    k_index = ...
        find(PredictedOrder == r,1,'first');


    %% --------------------------------------------------------
    % Correct theoretical slope
    %% --------------------------------------------------------

    % slope_ref = ...
    %     -r*log10(2);
  slope_ref = ...
        -r*log2(2);

    %% --------------------------------------------------------
    % Reference line starts near first numerical point
    %% --------------------------------------------------------

    Y0 = ...
        Y_plot(1,k_index) ...
        + reference_offset(ir);


    %% --------------------------------------------------------
    % Construct reference line
    %% --------------------------------------------------------

    Y_order = ...
        Y0 ...
        + slope_ref*(X_plot-X_plot(1));


    %% --------------------------------------------------------
    % Plot reference line
    %% --------------------------------------------------------

    plot( ...
        X_plot, ...
        Y_order, ...
        '--', ...
        'LineWidth',1.2, ...
        'DisplayName', ...
        sprintf('order %d',r));

end


%% ============================================================
% Axis formatting
%% ============================================================

ax = gca;

set(ax, ...
    'FontName','Times New Roman', ...
    'FontSize',11, ...
    'LineWidth',1);

grid on;

ax.GridLineStyle = ':';

ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';


%% ============================================================
% Axis labels
%% ============================================================

xlabel('number of time intervals in log scale', ...
    'Interpreter','tex', ...
    'FontName','Times New Roman', ...
    'FontSize',12);


ylabel('maximum error in log scale', ...
    'Interpreter','tex', ...
    'FontName','Times New Roman', ...
    'FontSize',12);


%% ============================================================
% Title
%
% p and q are italic
%% ============================================================

title_string = sprintf( ...
    '%s coarse ({\\it p} = %d) + %s fine ({\\it q} = %d)', ...
    method_short_name(coarse_method),p, ...
    method_short_name(fine_method),q);

title( ...
    title_string, ...
    'Interpreter','tex', ...
    'FontName','Times New Roman', ...
    'FontSize',12, ...
    'FontWeight','normal');


%% ============================================================
% Legend
%% ============================================================

lgd = legend( ...
    'Location','best', ...
    'FontSize',10, ...
    'Interpreter','tex');

set(lgd, ...
    'FontName','Times New Roman');


%% ============================================================
% Axis limits
%% ============================================================

x_margin = 0.25;

xlim([ ...
    min(X_plot)-x_margin, ...
    max(X_plot)+x_margin]);


%% ============================================================
% Optional x ticks
%
% We display log2(N), just like the reference paper displays
% logarithmically transformed degrees of freedom.
%% ============================================================

xticks(X_plot);

xticklabels( ...
    arrayfun(@(z) sprintf('%.2f',z), ...
    X_plot,'UniformOutput',false));


%% ============================================================
% Force Times New Roman
%% ============================================================

set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',11);


all_text = findall(gcf,'Type','text');

set(all_text, ...
    'FontName','Times New Roman');


set(gcf,'PaperPositionMode','auto');


%% ============================================================
% Local function:
% Construct one-step propagator
%% ============================================================

function stepper = build_stepper(L_h,dt,method)

    method = upper(strtrim(method));

    m = size(L_h,1);

    I = speye(m);


    switch method


        %% Backward Euler

        case 'BE'

            B = I-dt*L_h;

            Bdec = decomposition(B,'lu');

            stepper = @(U) ...
                Bdec\U;


        %% Implicit midpoint

        case 'IM2'

            B = I-0.5*dt*L_h;

            C = I+0.5*dt*L_h;

            Bdec = decomposition(B,'lu');

            stepper = @(U) ...
                Bdec\(C*U);


        %% Kutta RK3

        case 'RK3'

            stepper = @(U) ...
                step_RK3(U,L_h,dt);


        %% Classical RK4

        case 'RK4'

            stepper = @(U) ...
                step_RK4(U,L_h,dt);


        otherwise

            error(['Unknown method: %s\n' ...
                   'Available methods: BE, IM2, RK3, RK4'], ...
                   method);

    end

end


%% ============================================================
% Fine propagation over one coarse interval
%% ============================================================

function U = fine_propagation(U,Fstep,M)

    for j = 1:M

        U = Fstep(U);

    end

end


%% ============================================================
% Kutta third-order Runge-Kutta method
%% ============================================================

function Unew = step_RK3(U,L_h,dt)

    K1 = L_h*U;

    K2 = L_h*( ...
        U+0.5*dt*K1);

    K3 = L_h*( ...
        U-dt*K1+2*dt*K2);

    Unew = ...
        U+(dt/6)*( ...
        K1+4*K2+K3);

end


%% ============================================================
% Classical fourth-order Runge-Kutta method
%% ============================================================

function Unew = step_RK4(U,L_h,dt)

    K1 = L_h*U;

    K2 = L_h*( ...
        U+0.5*dt*K1);

    K3 = L_h*( ...
        U+0.5*dt*K2);

    K4 = L_h*( ...
        U+dt*K3);

    Unew = ...
        U+(dt/6)*( ...
        K1+2*K2+2*K3+K4);

end


%% ============================================================
% Formal order of method
%% ============================================================

function p = method_order(method)

    method = upper(strtrim(method));

    switch method

        case 'BE'

            p = 1;

        case 'IM2'

            p = 2;

        case 'RK3'

            p = 3;

        case 'RK4'

            p = 4;

        otherwise

            error('Unknown method: %s',method);

    end

end


%% ============================================================
% Full method name
%% ============================================================

function name = method_full_name(method)

    method = upper(strtrim(method));

    switch method

        case 'BE'

            name = 'Backward Euler';

        case 'IM2'

            name = 'Implicit midpoint';

        case 'RK3'

            name = 'Kutta RK3';

        case 'RK4'

            name = 'Classical RK4';

        otherwise

            error('Unknown method: %s',method);

    end

end


%% ============================================================
% Short method name
%% ============================================================

function name = method_short_name(method)

    method = upper(strtrim(method));

    switch method

        case 'BE'

            name = 'BE';

        case 'IM2'

            name = 'IM2';

        case 'RK3'

            name = 'RK3';

        case 'RK4'

            name = 'RK4';

        otherwise

            error('Unknown method: %s',method);

    end

end
