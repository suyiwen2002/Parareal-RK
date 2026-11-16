

clear; clc; close all;

%% 1. Physical parameters and parameter sweep
L = 1.0;
T = 1.0;

epsilonList = [0.10, 0.05, 0.02];
Jlist       = [40, 80, 160];

% Fixed temporal grid for a controlled parameter comparison
N     = 160;
Mfine = 10;
K     = 3;

coarseMethod = 'BE';
fineMethod   = 'RK4';

% Two independent RK4 reference resolutions, at the SAME coarse nodes.
% The finer reference below uses dt_ref = T/(N*MrefFine).
MrefCoarse = 40;
MrefFine   = 80;

% Relevant only if an implicit RK method is selected
newtonTol   = 1.0e-12;
newtonMaxit = 40;

% Serial propagation is sufficient for convergence verification.
% This experiment does NOT measure parallel speedup.
useParallel = false;

% Export one concise table containing 9 parameter combinations.
saveOutputs = true;
outputCSV   = 'AllenCahn_Epsilon_J_Sensitivity.csv';

%% 2. Methods and storage
pCoarse = methodOrder(coarseMethod);
qFine   = methodOrder(fineMethod);
assert(pCoarse <= qFine, 'Require p <= q.');

reaction      = @(u) u - u.^3;
reactionPrime = @(u) 1.0 - 3.0*u.^2;

numCases = numel(epsilonList)*numel(Jlist);
% [epsilon, J, N, M, E0, E1, E2, E3, FineError, ReferenceDifference]
Results = NaN(numCases, 10);

fprintf('\nAllen--Cahn epsilon/J sensitivity experiment\n');
fprintf('Coarse = %s, Fine = %s, N = %d, M = %d, K = %d\n', ...
        coarseMethod, fineMethod, N, Mfine, K);
fprintf('Reference RK4: %d and %d steps per coarse interval\n', ...
        MrefCoarse, MrefFine);

caseIndex = 0;
Tnodes = linspace(0, T, N+1);

%% 3. Loop over epsilon and spatial resolution J
for ie = 1:numel(epsilonList)
    epsilon = epsilonList(ie);
    diffusionCoeff = epsilon^2;

    for ij = 1:numel(Jlist)
        caseIndex = caseIndex + 1;
        J = Jlist(ij);
        h = L/J;

        %% 3.1 Reconstruct spatial semidiscretization
        x = linspace(0, L, J+1).';
        xInterior = x(2:end-1);
        ndof = J-1;

        e  = ones(ndof,1);
        A0 = spdiags([e, -2.0*e, e], -1:1, ndof, ndof);
        Ah = (diffusionCoeff/h^2)*A0;

        % Initial condition must be rebuilt on the current spatial grid.
        u0 = 0.8*sin(2.0*pi*xInterior);

        DeltaT = T/N;
        dt     = DeltaT/Mfine;
        dtRef1 = T/(N*MrefCoarse);
        dtRef2 = T/(N*MrefFine);

        % Exact spectral radius for the Dirichlet discrete diffusion operator.
        rhoAh = 4*diffusionCoeff/h^2*cos(pi/(2*J))^2;

        % Diffusion-based stability checks for explicit schemes.
        checkExplicitStability(coarseMethod, DeltaT, rhoAh, 'coarse');
        checkExplicitStability(fineMethod, dt, rhoAh, 'fine');
        checkExplicitStability('RK4', dtRef1, rhoAh, 'coarser reference');
        checkExplicitStability('RK4', dtRef2, rhoAh, 'finer reference');

        fprintf('\n[%d/%d] epsilon = %.3f, J = %d, h = %.5g\n', ...
                caseIndex, numCases, epsilon, J, h);

        %% 3.2 Independent high-accuracy RK4 reference for this (epsilon,J)
        Uref1 = serialRK4ReferenceAtNodes( ...
            Tnodes, u0, Ah, reaction, MrefCoarse);

        Uref2 = serialRK4ReferenceAtNodes( ...
            Tnodes, u0, Ah, reaction, MrefFine);

        % Measure the difference between the two reference resolutions.
        refDifference = max(rowGridNorm(Uref2-Uref1, h));

        %% 3.3 Serial fine propagator (for fine-error floor diagnostics)
        UfineSerial = serialFineTrajectory( ...
            u0, Ah, reaction, reactionPrime, T, N, Mfine, ...
            fineMethod, newtonTol, newtonMaxit);

        fineError = max(rowGridNorm(UfineSerial-Uref2, h));

        %% 3.4 Run the existing Parareal solver; no algorithm rewrite
        [Uhist, ~] = runPararealGeneral( ...
            u0, Ah, reaction, reactionPrime, T, N, Mfine, K, ...
            coarseMethod, fineMethod, useParallel, ...
            newtonTol, newtonMaxit);

        errors = NaN(1, K+1);
        for k = 0:K
            errors(k+1) = max( ...
                rowGridNorm(Uhist{k+1}-Uref2, h));
        end

        assert(all(isfinite(errors)), ...
            'Nonfinite Parareal error detected at epsilon=%g, J=%d.', ...
            epsilon, J);

        Results(caseIndex,:) = [ ...
            epsilon, J, N, Mfine, ...
            errors, fineError, refDifference];

        fprintf(' E(k=0..3)    : %.6e  %.6e  %.6e  %.6e\n', errors);
        fprintf(' Fine error   : %.6e\n', fineError);
        fprintf(' Ref. selfdiff: %.6e\n', refDifference);

        % The reference check is diagnostic, not a rigorous error bound.
        % If the finest iteration error approaches this scale, caution.
        if refDifference > 1.0e-10
            warning('Reference check exceeds 1e-10 at epsilon=%g, J=%d.', ...
                    epsilon, J);
        end
        if errors(end) <= 10*max(refDifference, eps)
            fprintf([' Note: k=%d error is near the reference/roundoff ' ...
                     'scale; do not overinterpret its digits.\n'], K);
        end
    end
end

%% 4. Display and export the 9-row experimental table
ResultTable = array2table(Results, 'VariableNames', { ...
    'epsilon','J','N','M', ...
    'E_k0','E_k1','E_k2','E_k3', ...
    'FineError','ReferenceDifference'});

fprintf('\n=============================================================\n');
fprintf('FINAL PARAMETER SENSITIVITY RESULTS\n');
fprintf('=============================================================\n');
disp(ResultTable);

if saveOutputs
    writetable(ResultTable, outputCSV);
    fprintf('Saved: %s\n', outputCSV);
end

fprintf(['\nInterpretation: E_k0...E_k3 compare each Parareal ' ...
         'iterate with its own\nhigh-accuracy semidiscrete RK4 ' ...
         'reference. They are NOT PDE spatial errors.\n']);

%% 5. Local functions copied from the original program

function p = methodOrder(method)
% Return formal order of selected method.

    method = ...
        upper(strtrim(method));

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

            error([ ...
                'Unknown method "%s". ', ...
                'Available methods: BE, IM2, RK3, RK4.' ...
                ], ...
                method);

    end

end

function name = methodFullName(method)
% Human-readable method name.

    method = ...
        upper(strtrim(method));

    switch method

        case 'BE'

            name = ...
                'Backward Euler';

        case 'IM2'

            name = ...
                'Implicit Midpoint';

        case 'RK3'

            name = ...
                'Classical RK3';

        case 'RK4'

            name = ...
                'Classical RK4';

        otherwise

            name = ...
                method;

    end

end

function checkExplicitStability( ...
    method,dt,rhoA,propagatorName)
% Simple diffusion-based explicit RK stability check.
%
% RK3 negative-real-axis stability limit:
%     approximately 2.5127
%
% RK4 negative-real-axis stability limit:
%     approximately 2.7853
%
% BE and IM2 are A-stable.

    method = ...
        upper(strtrim(method));

    value = ...
        dt*rhoA;

    switch method

        case 'RK3'

            limit = ...
                2.5127;

            if value >= limit

                warning([ ...
                    '%s RK3 propagator: dt*rho(-Ah) = %.6f ', ...
                    'is not below the approximate negative-real-axis ', ...
                    'stability limit %.4f.' ...
                    ], ...
                    propagatorName,value,limit);

            end

        case 'RK4'

            limit = ...
                2.7853;

            if value >= limit

                warning([ ...
                    '%s RK4 propagator: dt*rho(-Ah) = %.6f ', ...
                    'is not below the approximate negative-real-axis ', ...
                    'stability limit %.4f.' ...
                    ], ...
                    propagatorName,value,limit);

            end

        case {'BE','IM2'}

            % A-stable:
            % no corresponding diffusion CFL warning.

        otherwise

            error( ...
                'Unknown method.');

    end

end

function U = serialRK4ReferenceAtNodes( ...
    Tnodes,u0,A,f,Mref)
% High-accuracy serial classical RK4 reference solution.
%
% Each reference output interval is divided into exactly Mref RK4 steps.

    Tnodes = ...
        Tnodes(:);

    nOut = ...
        numel(Tnodes);

    ndof = ...
        numel(u0);

    U = ...
        zeros(nOut,ndof);

    U(1,:) = ...
        u0(:).';

    y = ...
        u0(:);

    for n = 1:nOut-1

        intervalLength = ...
            Tnodes(n+1)-Tnodes(n);

        dt = ...
            intervalLength/Mref;

        for m = 1:Mref

            y = ...
                rk4Step( ...
                y,A,f,dt);

        end

        U(n+1,:) = ...
            y.';

    end

end

function Usub = subsetReference( ...
    Uref,Nref,N)
% Extract nested coarse temporal nodes from reference trajectory.

    ratio = ...
        Nref/N;

    if abs(ratio-round(ratio)) > 1.0e-12

        error( ...
            'N must divide NrefGrid exactly.');

    end

    ratio = ...
        round(ratio);

    idx = ...
        1:ratio:(Nref+1);

    Usub = ...
        Uref(idx,:);

end

function U = serialFineTrajectory( ...
    u0,A,f,fp,T,N,Mfine,method,tol,maxit)
% Serial trajectory generated by selected fine propagator.

    ndof = ...
        numel(u0);

    U = ...
        zeros(N+1,ndof);

    U(1,:) = ...
        u0(:).';

    DeltaT = ...
        T/N;

    y = ...
        u0(:);

    for n = 1:N

        y = ...
            propagateOverInterval( ...
            y, ...
            A, ...
            f, ...
            fp, ...
            DeltaT, ...
            Mfine, ...
            method, ...
            tol, ...
            maxit);

        U(n+1,:) = ...
            y.';

    end

end

function [Uhist,timing] = runPararealGeneral( ...
    u0,A,f,fp,T,N,Mfine,K, ...
    coarseMethod,fineMethod,useParallel,tol,maxit)
% General Parareal:
%
%   U_{n+1}^{k+1}
%
%      = G(U_n^{k+1})
%        + F(U_n^k)
%        - G(U_n^k)
%
% G:
%   one coarseMethod step over DeltaT.
%
% F:
%   Mfine fineMethod steps over DeltaT.

    ndof = ...
        numel(u0);

    DeltaT = ...
        T/N;

    Uhist = ...
        cell(K+1,1);

    timing.initialCoarse = ...
        0.0;

    timing.fine = ...
        zeros(K,1);

    timing.correction = ...
        zeros(K,1);

    %% -------------------------------------------------------------------
    % Initial coarse prediction
    % --------------------------------------------------------------------

    timerInitial = tic;

    U0 = ...
        zeros(N+1,ndof);

    U0(1,:) = ...
        u0(:).';

    for n = 1:N

        v = ...
            U0(n,:).';

        y = ...
            oneStepMethod( ...
            v, ...
            A, ...
            f, ...
            fp, ...
            DeltaT, ...
            coarseMethod, ...
            tol, ...
            maxit);

        U0(n+1,:) = ...
            y.';

    end

    timing.initialCoarse = ...
        toc(timerInitial);

    Uhist{1} = ...
        U0;

    %% -------------------------------------------------------------------
    % Parareal iterations
    % --------------------------------------------------------------------

    for k = 0:K-1

        fprintf( ...
            'Parareal iteration %d / %d\n', ...
            k+1,K);

        Uold = ...
            Uhist{k+1};

        % ---------------------------------------------------------------
        % Fine propagation
        % ---------------------------------------------------------------

        fineOld = ...
            zeros(N,ndof);

        timerFine = ...
            tic;

        if useParallel && ...
                license('test','Distrib_Computing_Toolbox')

            parfor n = 1:N

                v = ...
                    Uold(n,:).';

                fineOld(n,:) = ...
                    propagateOverInterval( ...
                    v, ...
                    A, ...
                    f, ...
                    fp, ...
                    DeltaT, ...
                    Mfine, ...
                    fineMethod, ...
                    tol, ...
                    maxit).';

            end

        else

            for n = 1:N

                v = ...
                    Uold(n,:).';

                fineOld(n,:) = ...
                    propagateOverInterval( ...
                    v, ...
                    A, ...
                    f, ...
                    fp, ...
                    DeltaT, ...
                    Mfine, ...
                    fineMethod, ...
                    tol, ...
                    maxit).';

            end

        end

        timing.fine(k+1) = ...
            toc(timerFine);

        % ---------------------------------------------------------------
        % Sequential correction
        % ---------------------------------------------------------------

        timerCorrection = ...
            tic;

        Unew = ...
            zeros(N+1,ndof);

        Unew(1,:) = ...
            u0(:).';

        for n = 1:N

            vNew = ...
                Unew(n,:).';

            vOld = ...
                Uold(n,:).';

            Gnew = ...
                oneStepMethod( ...
                vNew, ...
                A, ...
                f, ...
                fp, ...
                DeltaT, ...
                coarseMethod, ...
                tol, ...
                maxit);

            Gold = ...
                oneStepMethod( ...
                vOld, ...
                A, ...
                f, ...
                fp, ...
                DeltaT, ...
                coarseMethod, ...
                tol, ...
                maxit);

            Unew(n+1,:) = ...
                ( ...
                Gnew ...
                + fineOld(n,:).' ...
                - Gold ...
                ).';

        end

        timing.correction(k+1) = ...
            toc(timerCorrection);

        Uhist{k+2} = ...
            Unew;

    end

end

function y = propagateOverInterval( ...
    v,A,f,fp,DeltaT,M,method,tol,maxit)
% Apply M steps of selected method on one coarse interval.

    dt = ...
        DeltaT/M;

    y = ...
        v;

    for m = 1:M

        y = ...
            oneStepMethod( ...
            y, ...
            A, ...
            f, ...
            fp, ...
            dt, ...
            method, ...
            tol, ...
            maxit);

    end

end

function y = oneStepMethod( ...
    v,A,f,fp,dt,method,tol,maxit)
% One time step of selected method.

    method = ...
        upper(strtrim(method));

    switch method

        case 'BE'

            y = ...
                backwardEulerStep( ...
                v,A,f,fp,dt,tol,maxit);

        case 'IM2'

            y = ...
                implicitMidpointStep( ...
                v,A,f,fp,dt,tol,maxit);

        case 'RK3'

            y = ...
                rk3Step( ...
                v,A,f,dt);

        case 'RK4'

            y = ...
                rk4Step( ...
                v,A,f,dt);

        otherwise

            error( ...
                'Unknown time-integration method.');

    end

end

function y = backwardEulerStep( ...
    v,A,f,fp,dt,tol,maxit)
% Backward Euler:
%
%   y = v + dt*(A*y + f(y))
%
% Formal order = 1.
%
% Newton iteration is used.

    y = ...
        v;

    n = ...
        numel(v);

    I = ...
        speye(n);

    for it = 1:maxit

        residual = ...
            y ...
            - v ...
            - dt*( ...
            A*y ...
            + f(y));

        if norm(residual,inf) <= ...
                tol*(1.0+norm(y,inf))

            return;

        end

        JNewton = ...
            I ...
            - dt*( ...
            A ...
            + spdiags( ...
            fp(y), ...
            0, ...
            n, ...
            n));

        delta = ...
            -JNewton\residual;

        y = ...
            y+delta;

        if norm(delta,inf) <= ...
                tol*(1.0+norm(y,inf))

            return;

        end

    end

    warning( ...
        'Backward Euler Newton iteration reached maxit = %d.', ...
        maxit);

end

function y = implicitMidpointStep( ...
    v,A,f,fp,dt,tol,maxit)
% Implicit midpoint:
%
%   y = v
%       + dt * [
%
%           A*((v+y)/2)
%           + f((v+y)/2)
%
%         ]
%
% Formal order = 2.

    y = ...
        v;

    n = ...
        numel(v);

    I = ...
        speye(n);

    for it = 1:maxit

        ymid = ...
            0.5*(v+y);

        residual = ...
            y ...
            - v ...
            - dt*( ...
            A*ymid ...
            + f(ymid));

        if norm(residual,inf) <= ...
                tol*(1.0+norm(y,inf))

            return;

        end

        JNewton = ...
            I ...
            - 0.5*dt*( ...
            A ...
            + spdiags( ...
            fp(ymid), ...
            0, ...
            n, ...
            n));

        delta = ...
            -JNewton\residual;

        y = ...
            y+delta;

        if norm(delta,inf) <= ...
                tol*(1.0+norm(y,inf))

            return;

        end

    end

    warning( ...
        'Implicit midpoint Newton iteration reached maxit = %d.', ...
        maxit);

end

function ynew = rk3Step( ...
    y,A,f,dt)
% Classical third-order Kutta Runge-Kutta method.
%
% Butcher tableau:
%
%       0 |
%     1/2 |  1/2
%       1 | -1     2
%   --------------------
%         | 1/6   2/3   1/6
%
% Formal order = 3.

    k1 = ...
        A*y ...
        + f(y);

    y2 = ...
        y ...
        + 0.5*dt*k1;

    k2 = ...
        A*y2 ...
        + f(y2);

    y3 = ...
        y ...
        - dt*k1 ...
        + 2.0*dt*k2;

    k3 = ...
        A*y3 ...
        + f(y3);

    ynew = ...
        y ...
        + (dt/6.0)*( ...
        k1 ...
        + 4.0*k2 ...
        + k3);

end

function ynew = rk4Step( ...
    y,A,f,dt)
% Classical fourth-order Runge-Kutta method.
%
% Formal order = 4.

    k1 = ...
        A*y ...
        + f(y);

    y2 = ...
        y ...
        + 0.5*dt*k1;

    k2 = ...
        A*y2 ...
        + f(y2);

    y3 = ...
        y ...
        + 0.5*dt*k2;

    k3 = ...
        A*y3 ...
        + f(y3);

    y4 = ...
        y ...
        + dt*k3;

    k4 = ...
        A*y4 ...
        + f(y4);

    ynew = ...
        y ...
        + (dt/6.0)*( ...
        k1 ...
        + 2.0*k2 ...
        + 2.0*k3 ...
        + k4);

end

function nr = rowGridNorm(X,h)
% Discrete L2 norm:
%
%   ||v||_h = sqrt(h*sum_j |v_j|^2).

    nr = ...
        sqrt( ...
        h*sum(abs(X).^2,2));

end
