
clear; clc;
format long e;



%% 1. Problem parameters
d     = 0.01;
alpha = 1.0;
T     = 1.0;

J_list = [20, 40, 80, 160, 320];
K = 3;

numJ = length(J_list);

%% 2. Allocate error arrays
Espace = zeros(numJ, 1);
Etime  = zeros(numJ, K+1);
Efull  = zeros(numJ, K+1);

rSpace = NaN(numJ, 1);
rTime  = NaN(numJ, K+1);
rFull  = NaN(numJ, K+1);

%% ============================================================
% 3. Joint refinement loop
%% ============================================================
for iJ = 1:numJ

    %% Joint mesh parameters
    J = J_list(iJ);
    N = J;
    M = J;

    h      = 1/J;
    DeltaT = T/N;
    dt     = DeltaT/M;

    fprintf('\nJ=%d, N=%d, M=%d, h=%.6e, dt=%.6e\n', ...
            J, N, M, h, dt);

    %% 3.1 Spatial grid
    x = (1:J-1)' * h;
    m = J - 1;

    %% 3.2 Spatial finite-difference matrix
    e = ones(m,1);

    A_h = (d/h^2) * ...
        spdiags([e, -2*e, e], -1:1, m, m);

    L_h = A_h + alpha * speye(m);

    %% 3.3 Initial condition
    v1 = sin(pi*x);
    v2 = sin(2*pi*x);

    U0 = v1 + 0.5*v2;

    %% 3.4 Exact semidiscrete eigenvalues
    mu1_h = 4/h^2 * sin(pi*h/2)^2;
    mu2_h = 4/h^2 * sin(pi*h)^2;

    lambda1_h = alpha - d*mu1_h;
    lambda2_h = alpha - d*mu2_h;

    %% 3.5 Exact solutions at coarse time nodes
    tNodes = (0:N) * DeltaT;

    % Exact semidiscrete solution U_h(t)
    UhRef = ...
        v1 * exp(lambda1_h*tNodes) ...
        + 0.5*v2 * exp(lambda2_h*tNodes);

    % Exact continuous PDE solution restricted to grid
    PuRef = ...
        v1 * exp((alpha-d*pi^2)*tNodes) ...
        + 0.5*v2 * exp((alpha-4*d*pi^2)*tNodes);

    %% 3.6 Spatial discretization error
    spatialDiff = PuRef - UhRef;

    spatialNorm = sqrt( ...
        h * sum(spatialDiff.^2, 1));

    Espace(iJ) = max(spatialNorm);

    %% 3.7 Check fine RK4 diffusion stability
    lambda_min = alpha ...
        - 4*d/h^2 * cos(pi/(2*J))^2;

    if dt*lambda_min < -2.785
        warning('RK4 stability limit may be violated for J=%d.', J);
    end

    %% 3.8 Backward Euler coarse propagator
    B = speye(m) - DeltaT*L_h;

    Bdec = decomposition(B, 'lu');

    %% 3.9 Allocate Parareal solutions
    % U(:,n+1,k+1) corresponds to U_n^(k).
    U = zeros(m, N+1, K+1);

    %% Initial coarse prediction: k = 0
    U(:,1,1) = U0;

    for n = 1:N
        U(:,n+1,1) = Bdec \ U(:,n,1);
    end

    %% 3.10 Parareal correction iterations
    for k = 1:K

        U(:,1,k+1) = U0;

        for n = 1:N

            % New coarse propagation
            G_new = Bdec \ U(:,n,k+1);

            % Fine propagation from previous iterate
            F_old = fine_RK4( ...
                U(:,n,k), L_h, dt, M);

            % Old coarse propagation
            G_old = Bdec \ U(:,n,k);

            % Parareal correction
            U(:,n+1,k+1) = ...
                G_new + (F_old - G_old);

        end
    end

    %% 3.11 Temporal and fully discrete errors
    for k = 0:K

        % Temporal error: compared with U_h
        timeDiff = U(:,:,k+1) - UhRef;

        timeNorm = sqrt( ...
            h * sum(timeDiff.^2, 1));

        Etime(iJ,k+1) = max(timeNorm);

        % Fully discrete error: compared with P_h u
        fullDiff = U(:,:,k+1) - PuRef;

        fullNorm = sqrt( ...
            h * sum(fullDiff.^2, 1));

        Efull(iJ,k+1) = max(fullNorm);

    end

    fprintf('Completed J=%d, Espace=%.8e\n', ...
            J, Espace(iJ));

end

%% ============================================================
% 4. Observed convergence orders
%% ============================================================
for iJ = 2:numJ

    % Actual spatial mesh ratio
    ratio = J_list(iJ)/J_list(iJ-1);

    rSpace(iJ) = ...
        log(Espace(iJ-1)/Espace(iJ)) / log(ratio);

    for k = 0:K

        rTime(iJ,k+1) = ...
            log(Etime(iJ-1,k+1)/Etime(iJ,k+1)) ...
            / log(ratio);

        rFull(iJ,k+1) = ...
            log(Efull(iJ-1,k+1)/Efull(iJ,k+1)) ...
            / log(ratio);

    end
end

%% ============================================================
% 5. Print Table 3: joint refinement results
%% ============================================================
fprintf('\n');
fprintf('JOINT SPATIAL-TEMPORAL REFINEMENT RESULTS\n');
fprintf('BE coarse (p=1), RK4 fine (q=4), N=M=J\n\n');

fprintf('%5s %11s %11s %11s %3s | %12s %12s | %7s %7s\n', ...
        'J','h','DeltaT','dt','k', ...
        'E_time','E_full','r_time','r_full');

fprintf('%s\n', repmat('-',1,102));

% Record results for export
Rows = NaN(numJ*(K+1), 9);
rowId = 0;

for k = 0:K

    for iJ = 1:numJ

        J = J_list(iJ);
        h = 1/J;
        N = J;
        M = J;

        DeltaT = T/N;
        dt = DeltaT/M;

        fprintf('%5d %11.4e %11.4e %11.4e %3d | %12.5e %12.5e | %7.3f %7.3f\n', ...
            J, h, DeltaT, dt, k, ...
            Etime(iJ,k+1), Efull(iJ,k+1), ...
            rTime(iJ,k+1), rFull(iJ,k+1));

        rowId = rowId + 1;

        Rows(rowId,:) = [ ...
            J, h, DeltaT, dt, k, ...
            Etime(iJ,k+1), Efull(iJ,k+1), ...
            rTime(iJ,k+1), rFull(iJ,k+1)];

    end

    fprintf('%s\n', repmat('-',1,102));
end

%% ============================================================
% 6. Print spatial errors for verification
%% ============================================================
fprintf('\nSPATIAL ERROR CHECK\n');

fprintf('%6s %16s %10s\n', ...
        'J','E_space','r_space');

for iJ = 1:numJ
    fprintf('%6d %16.8e %10.4f\n', ...
        J_list(iJ), Espace(iJ), rSpace(iJ));
end

%% ============================================================
% 7. Export Table 3 to CSV
%% ============================================================
JointTable = array2table(Rows, ...
    'VariableNames', { ...
    'J','h','DeltaT','dt','k', ...
    'E_time','E_full','r_time','r_full'});

writetable(JointTable, 'JointRefinement_Table3.csv');

fprintf('\nResults saved to JointRefinement_Table3.csv\n');

%% ============================================================
% 8. Local function: Classical RK4
%% ============================================================
function U = fine_RK4(U, L_h, dt, M)

    for j = 1:M

        K1 = L_h * U;
        K2 = L_h * (U + 0.5*dt*K1);
        K3 = L_h * (U + 0.5*dt*K2);
        K4 = L_h * (U + dt*K3);

        U = U + (dt/6) * ...
            (K1 + 2*K2 + 2*K3 + K4);

    end

end
