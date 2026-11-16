function FisherKPP2D_Competitive_Benchmarks()


%% ======================== USER CONFIGURATION ===========================
L = 1.0;
T = 5.0;
D = 1.0e-2;
r = 1.0;
alpha = 100.0;
J = 160;                     % (J-1)^2 = 25281 unknowns

N = 128;                     % Parareal coarse intervals, fixed
M = 500;                     % RK4 fine steps per coarse interval
P = 32;                      % number of MATLAB parallel workers
Klist = [1 2 3];             % test Parareal corrections
nRepeats = 3;                % independent timing repeats

tolerances = [1e-3 1e-4 1e-5];
serialStepCandidates = [20 40 80 160 320 640 1280 2560 ...
                        5120 10240 20480 40960];

% Reference: serial RK4 with 40,000 and 80,000 steps.
Nref = 40;
MrefCoarse = 1000;
MrefFine = 2000;

% Nonlinear implicit solve parameters (for BE).
fpTol = 1e-10;
fpMaxit = 100;
usePredictor = true;

% Exclude parallel pool startup and warm-up from all timing statistics.
saveResults = true;
% ========================================================================

assert(P>=1 && P<=N && floor(P)==P, 'Require integer 1 <= P <= N.');
assert(all(diff(serialStepCandidates)>0), ...
       'serialStepCandidates must be strictly increasing.');
assert(all(tolerances>0), 'Tolerances must be positive.');
assert(all(Klist>=1) && all(diff(Klist)>0), ...
       'Klist must be strictly increasing positive integers.');
assert(nRepeats>=2, 'At least two repetitions are required for SD.');

% Reproduce the original benchmark's single-threaded client setting.
originalThreads = maxNumCompThreads;
maxNumCompThreads(1);
threadCleanup = onCleanup(@() maxNumCompThreads(originalThreads)); %#ok<NASGU>

%% ======================= SPATIAL DISCRETIZATION =========================
h = L/J;
x = (1:J-1)'*h;
[X,Y] = meshgrid(x,x);
u0 = exp(-alpha*((X-L/2).^2+(Y-L/2).^2));
u0 = u0(:);
ndof = numel(u0);
e = ones(J-1,1);
T1 = spdiags([e -2*e e],[-1 0 1],J-1,J-1);
A = (D/h^2)*(kron(speye(J-1),T1)+kron(T1,speye(J-1)));
I = speye(ndof);

% Negative-real-axis stability test for RK4.
rhoDiff = 8*D/h^2*cos(pi/(2*J))^2;
fineDt = T/(N*M);
refDt1 = T/(Nref*MrefCoarse);
refDt2 = T/(Nref*MrefFine);
assert(max([fineDt refDt1 refDt2])*rhoDiff < 2.785, ...
       'RK4 stability restriction violated. Increase fine steps.');

fprintf('\n============================================================\n');
fprintf('2D FISHER-KPP: ACCURACY-MATCHED COMPETITIVE BASELINES\n');
fprintf('J=%d, ndof=%d, T=%g, D=%g, r=%g\n', J,ndof,T,D,r);
fprintf('Parareal N=%d, M=%d, P=%d, K=%s\n', ...
        N,M,P,mat2str(Klist));
fprintf('Tolerances=%s; repeats=%d\n',mat2str(tolerances),nRepeats);
fprintf('RK4 dt=%.7g; diffusion stability number=%.5g\n', ...
        fineDt, fineDt*rhoDiff);
fprintf('All reported timings EXCLUDE LU, reference and pool startup.\n');
fprintf('============================================================\n');

%% =================== SHARED HIGH-ACCURACY REFERENCE =====================
fprintf('\nCalculating independent serial RK4 references...\n');
yRef1 = serialRK4Final(u0,A,r,T,Nref,MrefCoarse);
yRef  = serialRK4Final(u0,A,r,T,Nref,MrefFine);
refDifference = h*norm(yRef-yRef1,2);
fprintf('Reference self-difference (grid L2) = %.12e\n',refDifference);
if refDifference > min(tolerances)/100
    error(['Reference check is not small enough for the requested ', ...
           'tolerance. Increase MrefFine and MrefCoarse.']);
end

%% ========================== OUTPUT STORAGE =============================
% Config fields: Method, Nsteps, K, Workers, ErrorFinal,
% Tmean, Tsd, InitialMean, InitialSD, FineMean, FineSD,
% CorrectionMean, CorrectionSD, OtherMean, OtherSD, LUSetupSeconds.
configColumns = {'Method','Nsteps','K','Workers','ErrorFinal', ...
    'Tmean_s','Tsd_s','InitialMean_s','InitialSD_s', ...
    'FineMean_s','FineSD_s','CorrectionMean_s','CorrectionSD_s', ...
    'OtherMean_s','OtherSD_s','LUSetup_s'};
configs = cell(0,numel(configColumns));

rawColumns = {'Method','Nsteps','K','Workers','Repeat', ...
              'Total_s','Initial_s','Fine_s','Correction_s','Other_s'};
raw = cell(0,numel(rawColumns));

screenColumns = {'Method','Nsteps','dt','ErrorFinal','SolveSucceeded'};
screening = cell(0,numel(screenColumns));

%% ============================ SERIAL BE =================================
serialMethods = {'BE'};

for im = 1:numel(serialMethods)
    method = serialMethods{im};
    label = ['Serial ' method];
    fprintf('\n=======================================================\n');
    fprintf('SCREENING %s AGAINST SAME REFERENCE\n',label);
    fprintf('=======================================================\n');

    % First find a qualifying N for every prescribed tolerance.
    candidateN = [];
    candidateError = [];

    for ii = 1:numel(serialStepCandidates)
        ns = serialStepCandidates(ii);
        dt = T/ns;
        try
            [solver,~] = prepareImplicitSolver(A,I,dt);
            y = serialImplicitFinal(u0,A,r,T,ns,solver, ...
                                    fpTol,fpMaxit,usePredictor);
            E = h*norm(y-yRef,2);
            succeeded = all(isfinite(y)) && isfinite(E);
            if ~succeeded
                E = inf;
            end
        catch ME
            warning('%s, ns=%d: %s',label,ns,ME.message);
            E = inf;
            succeeded = false;
        end

        screening(end+1,:) = {label,ns,dt,E,succeeded}; %#ok<AGROW>
        fprintf(' %-10s ns=%6d  dt=%9.3e  E=%.7e\n',label,ns,dt,E);
        candidateN(end+1) = ns; %#ok<AGROW>
        candidateError(end+1) = E; %#ok<AGROW>

        if E <= min(tolerances)
            break;
        end
    end

    % For each tolerance, use the coarsest qualifying tested serial grid.
    chosenN = [];
    for it = 1:numel(tolerances)
        idx = find(candidateError<=tolerances(it),1,'first');
        if isempty(idx)
            warning('%s never achieved tolerance %g.',label,tolerances(it));
        else
            chosenN(end+1) = candidateN(idx); %#ok<AGROW>
        end
    end
    chosenN = unique(chosenN,'stable');

    % Benchmark each distinct chosen serial grid after one warm-up.
    for ns = chosenN
        dt = T/ns;
        [solver,tLU] = prepareImplicitSolver(A,I,dt);

        warm = serialImplicitFinal(u0,A,r,T,ns,solver, ...
                                   fpTol,fpMaxit,usePredictor); %#ok<NASGU>
        times = zeros(nRepeats,1);
        for rep = 1:nRepeats
            tt = tic;
            y = serialImplicitFinal(u0,A,r,T,ns,solver, ...
                                    fpTol,fpMaxit,usePredictor);
            times(rep) = toc(tt);
            raw(end+1,:) = {label,ns,NaN,1,rep, ...
                            times(rep),NaN,NaN,NaN,NaN}; %#ok<AGROW>
        end
        E = h*norm(y-yRef,2);
        configs(end+1,:) = {label,ns,NaN,1,E, ...
            mean(times),std(times), ...
            NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,tLU}; %#ok<AGROW>

        fprintf('TIMED %-10s ns=%d: E=%.6e, T=%.6f +/- %.6f s (LU %.4fs excluded)\n', ...
            label,ns,E,mean(times),std(times),tLU);
    end
end

%% ======================== PARALLEL POOL SETUP ===========================
% This step is deliberately outside all measured runs.
useParallel = (P > 1);
if useParallel
    if ~license('test','Distrib_Computing_Toolbox')
        error('Parallel Computing Toolbox required for P>1; set P=1 to test serially.');
    end
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= P
        if ~isempty(pool)
            delete(pool);
        end
        fprintf('Starting parallel pool with %d workers (not timed)...\n',P);
        pool = parpool('local',P);
    end
    assert(pool.NumWorkers == P);
    fprintf('Warming all workers with RK4 tasks (not timed)...\n');
    warmFine = zeros(P,1);
    warmDT = T/N;
    parfor iw = 1:P
        tmp = fineRK4Interval(u0,A,r,warmDT,M);
        warmFine(iw) = tmp(1);
    end
    assert(all(isfinite(warmFine)));
else
    fprintf('P=1: parfor disabled; results are for serial debugging.\n');
end

%% ======================= PARAREAL BE-RK4 ================================
pararealCoarse = {'BE'};

for im = 1:numel(pararealCoarse)
    method = pararealCoarse{im};
    label = ['Parareal ' method '-RK4'];
    fprintf('\n=======================================================\n');
    fprintf('TIMING %s\n',label);
    fprintf('=======================================================\n');

    [Gsolver,tLU] = prepareImplicitSolver(A,I,T/N);

    % Path warm-up; a run with K=1 initializes the Parareal code path.
    [~,~] = runParareal(u0,A,r,T,N,M,1,Gsolver, ...
                       fpTol,fpMaxit,usePredictor,useParallel);

    for K = Klist
        totalTime = zeros(nRepeats,1);
        initialTime = zeros(nRepeats,1);
        fineTime = zeros(nRepeats,1);
        correctionTime = zeros(nRepeats,1);
        otherTime = zeros(nRepeats,1);

        for rep = 1:nRepeats
            tt = tic;
            [Ufinal,timing] = runParareal(u0,A,r,T,N,M,K, ...
                           Gsolver,fpTol,fpMaxit,usePredictor,useParallel);
            totalTime(rep) = toc(tt);
            initialTime(rep) = timing.initialCoarse;
            fineTime(rep) = sum(timing.fine);
            correctionTime(rep) = sum(timing.correction);
            otherTime(rep) = totalTime(rep)-initialTime(rep) ...
                            -fineTime(rep)-correctionTime(rep);
            raw(end+1,:) = {label,N,K,P,rep,totalTime(rep), ...
                initialTime(rep),fineTime(rep),correctionTime(rep), ...
                otherTime(rep)}; %#ok<AGROW>
        end

        % yRef is the common terminal-time reference, not an initial state.
        E = h*norm(Ufinal-yRef,2);

        configs(end+1,:) = {label,N,K,P,E, ...
            mean(totalTime),std(totalTime), ...
            mean(initialTime),std(initialTime), ...
            mean(fineTime),std(fineTime), ...
            mean(correctionTime),std(correctionTime), ...
            mean(otherTime),std(otherTime),tLU}; %#ok<AGROW>

        fprintf('K=%d: E=%.7e, T=%.6f +/- %.6f s', ...
                K,E,mean(totalTime),std(totalTime));
        fprintf(', initial=%.3f, fine=%.3f, correction=%.3f s\n', ...
                mean(initialTime),mean(fineTime),mean(correctionTime));
    end
end

%% ======================= ACCURACY-MATCHED SELECTION =====================
ConfigTable = cell2table(configs,'VariableNames',configColumns);
RawTimingTable = cell2table(raw,'VariableNames',rawColumns);
ScreeningTable = cell2table(screening,'VariableNames',screenColumns);

% Force numeric/logical column types (robust across MATLAB releases).
for c = 2:numel(configColumns)
    ConfigTable.(configColumns{c}) = cell2mat(configs(:,c));
end
for c = 2:numel(rawColumns)
    RawTimingTable.(rawColumns{c}) = cell2mat(raw(:,c));
end
for c = 2:numel(screenColumns)
    ScreeningTable.(screenColumns{c}) = cell2mat(screening(:,c));
end

allMethods = {'Serial BE','Parareal BE-RK4'};
bestColumns = {'Tolerance','Method','Nsteps','K','Workers', ...
               'ErrorFinal','TimeMean_s','TimeSD_s','Qualified'};
bestRows = cell(0,numel(bestColumns));

for it = 1:numel(tolerances)
    tol = tolerances(it);
    for im = 1:numel(allMethods)
        method = allMethods{im};
        inds = find(strcmp(ConfigTable.Method,method) & ...
                    ConfigTable.ErrorFinal<=tol);

        if isempty(inds)
            bestRows(end+1,:) = {tol,method,NaN,NaN,NaN, ...
                                NaN,NaN,NaN,false}; %#ok<AGROW>
            continue
        end
        % Fastest measured qualifying configuration among those tested.
        [~,localIdx] = min(ConfigTable.Tmean_s(inds));
        idx = inds(localIdx);
        bestRows(end+1,:) = {tol,method,ConfigTable.Nsteps(idx), ...
            ConfigTable.K(idx),ConfigTable.Workers(idx), ...
            ConfigTable.ErrorFinal(idx),ConfigTable.Tmean_s(idx), ...
            ConfigTable.Tsd_s(idx),true}; %#ok<AGROW>
    end
end

BestTable = cell2table(bestRows,'VariableNames',bestColumns);
BestTable.Tolerance = cell2mat(bestRows(:,1));
for c = 3:numel(bestColumns)
    BestTable.(bestColumns{c}) = cell2mat(bestRows(:,c));
end
BestTable.SpeedupVsSerialBE = NaN(height(BestTable),1);

for it = 1:numel(tolerances)
    tol = tolerances(it);
    idxBE = find(BestTable.Tolerance==tol & ...
                 strcmp(BestTable.Method,'Serial BE'),1);
    for ir = find(BestTable.Tolerance==tol).'
        if BestTable.Qualified(ir)
            if ~isempty(idxBE) && BestTable.Qualified(idxBE)
                BestTable.SpeedupVsSerialBE(ir) = ...
                    BestTable.TimeMean_s(idxBE)/BestTable.TimeMean_s(ir);
            end
        end
    end
end

fprintf('\n============================================================\n');
fprintf('ACCURACY-MATCHED COMPETITIVE COMPARISON\n');
fprintf('Error metric: ||U(T)-Uref(T)||_h; same J and same reference.\n');
fprintf('All times in seconds: sample mean and sample SD (%d repeats).\n',nRepeats);
fprintf('============================================================\n');
disp(BestTable);

fprintf('\nCOMPLETE TIMED CONFIGURATIONS\n');
disp(ConfigTable);

if saveResults
    writetable(BestTable,      'FisherKPP2D_Competitive_AccuracyMatched.csv');
    writetable(ConfigTable,    'FisherKPP2D_Competitive_TimedConfigs.csv');
    writetable(RawTimingTable, 'FisherKPP2D_Competitive_RawTimings.csv');
    writetable(ScreeningTable, 'FisherKPP2D_Competitive_SerialScreening.csv');
    fprintf('\nFour CSV files saved to current MATLAB working directory.\n');
end

fprintf('\nAll benchmarks exclude LU setup, RK4 reference, warm-up, and pool startup.\n');
fprintf('LU setup time is listed separately in TimedConfigs.csv.\n');
fprintf('A reference self-difference is a diagnostic, not a rigorous error bound.\n');

end  % end of main function


%% ============================== FUNCTIONS ==============================

function [solver,tSetup] = prepareImplicitSolver(A,I,dt)
% Reusable sparse LU factorization of diffusion part of BE equation.
    B = I-dt*A;
    tt = tic;
    solver = decomposition(B,'lu');
    tSetup = toc(tt);
end

function y = implicitStepFisher(v,A,r,dt,solver,tol,maxit,predict)
% Fully implicit reaction + diffusion with Picard fixed-point iterations.
% BE: (I-dt*A)y = v + dt*r*y*(1-y).
% The iteration uses the exact ORIGINAL nonlinear residual as criterion.
    v = v(:);
    fv = r*v.*(1-v);

    if predict
        y = v+dt*(A*v+fv);
        if any(~isfinite(y))
            y = v;
        end
    else
        y = v;
    end

    rhsFixed = v;
    diffusionWeight = dt;
    reactionWeight = dt;

    for it = 0:maxit
        fy = r*y.*(1-y);
        R = y-rhsFixed-diffusionWeight*(A*y)-reactionWeight*fy;
        if norm(R,inf)<=tol
            return;
        end
        if it==maxit
            error('BE Picard failed, dt=%.5g, residual=%.4e', ...
                  dt,norm(R,inf));
        end
        y = solver\(rhsFixed+reactionWeight*fy);
        if any(~isfinite(y))
            error('BE Picard produced nonfinite values.');
        end
    end
end

function y = serialImplicitFinal(u0,A,r,T,ns,solver,tol,maxit,predict)
% Compute terminal value only (same discrete grid as Parareal).
    dt = T/ns;
    y = u0;
    for n = 1:ns
        y = implicitStepFisher(y,A,r,dt,solver,tol,maxit,predict);
    end
end

function y = rk4StepFisher(y,A,r,dt)
    k1 = A*y + r*y.*(1-y);
    y2 = y+0.5*dt*k1;
    k2 = A*y2 + r*y2.*(1-y2);
    y3 = y+0.5*dt*k2;
    k3 = A*y3 + r*y3.*(1-y3);
    y4 = y+dt*k3;
    k4 = A*y4 + r*y4.*(1-y4);
    y = y+(dt/6)*(k1+2*k2+2*k3+k4);
end

function y = fineRK4Interval(v,A,r,DeltaT,M)
    dt = DeltaT/M;
    y = v(:);
    for mm = 1:M
        y = rk4StepFisher(y,A,r,dt);
    end
    if any(~isfinite(y))
        error('Nonfinite RK4 result.');
    end
end

function y = serialRK4Final(u0,A,r,T,ns,m)
    dt = T/(ns*m);
    y = u0(:);
    for ii = 1:(ns*m)
        y = rk4StepFisher(y,A,r,dt);
    end
    if any(~isfinite(y))
        error('Nonfinite reference solution.');
    end
end

function [uFinal,timing] = runParareal(u0,A,r,T,N,M,K, ...
                             coarseSolver,tol,maxit,predict,useParallel)
% Cached-coarse Parareal for a single fixed time partition.
% U_{n+1}^{k+1} = G(U_n^{k+1}) + F(U_n^k) - G(U_n^k).
    DeltaT = T/N;
    ndof = numel(u0);

    timing.initialCoarse = 0;
    timing.fine = zeros(K,1);
    timing.correction = zeros(K,1);

    % Sequential initial coarse prediction and saved coarse evaluations.
    tt = tic;
    Uold = zeros(N+1,ndof);
    Uold(1,:) = u0(:).';
    GoldStored = zeros(N,ndof);
    for n = 1:N
        v = Uold(n,:).';
        y = implicitStepFisher(v,A,r,DeltaT,coarseSolver,tol,maxit,predict);
        Uold(n+1,:) = y.';
        GoldStored(n,:) = y.';
    end
    timing.initialCoarse = toc(tt);

    for kk = 1:K
        fineOld = zeros(N,ndof);
        ttFine = tic;
        if useParallel
            parfor n = 1:N
                vOld = Uold(n,:).';
                yFine = fineRK4Interval(vOld,A,r,DeltaT,M);
                fineOld(n,:) = yFine.';
            end
        else
            for n = 1:N
                vOld = Uold(n,:).';
                yFine = fineRK4Interval(vOld,A,r,DeltaT,M);
                fineOld(n,:) = yFine.';
            end
        end
        timing.fine(kk) = toc(ttFine);

        ttCorr = tic;
        Unew = zeros(N+1,ndof);
        Unew(1,:) = u0(:).';
        GnewStored = zeros(N,ndof);
        for n = 1:N
            vNew = Unew(n,:).';
            Gnew = implicitStepFisher(vNew,A,r,DeltaT,coarseSolver,tol,maxit,predict);
            Unew(n+1,:) = (Gnew+fineOld(n,:).'-GoldStored(n,:).').';
            GnewStored(n,:) = Gnew.';
            if any(~isfinite(Unew(n+1,:)))
                error('Nonfinite Parareal correction.');
            end
        end
        timing.correction(kk) = toc(ttCorr);
        Uold = Unew;
        GoldStored = GnewStored;
    end

    uFinal = Uold(end,:).';
end
