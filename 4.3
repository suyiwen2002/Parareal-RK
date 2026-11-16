function FisherKPP2D_Merged_Scaling_Tolerance()


%% ============================== SETTINGS ================================
L = 1.0; T = 5.0; D = 1.0e-2; r = 1.0;
alpha = 100.0;

J = 160;
N = 128;
M = 500;
Plist = [6 8 16 32 64];
Kmax = 3;
Klist = 1:Kmax;
nRepeats = 5;
TolList = [1e-3 1e-4 1e-5 1e-6 1e-7];

% Independent RK4 semidiscrete references, 40k and 80k time steps.
Nref = 40;
MrefCoarse = 1000;
MrefFine = 2000;

% Accuracy-selected SERIAL RK4: Nserial*Mserial total RK4 steps.
% The first stability-qualified candidate reaching each tau is selected.
Nserial = 10;
MserialCandidates = [50 100 200 300 400 500 800 1000 1500 2000 4000 6400];

% Retain the supplied BE coarse solver parameters.
fpTol = 1e-10;
fpMaxit = 100;
usePredictor = true;

saveResults = true;
makePlots = true;
% ========================================================================

assert(all(Plist >= 1 & mod(Plist,1)==0) & all(Plist<=N), ...
    'Require integer 1<=P<=N. The current experiment uses fixed N=128.');
assert(numel(unique(Plist))==numel(Plist),'Plist has duplicates.');
assert(Kmax>=1 && mod(Kmax,1)==0,'Kmax must be a positive integer.');
assert(nRepeats>=2,'nRepeats must be >=2 for sample SD.');
assert(all(TolList>0),'All tolerances must be positive.');
assert(all(diff(MserialCandidates)>0), ...
    'MserialCandidates must be strictly increasing.');

% Single client thread; all serial timings are taken before pool creation.
oldThreads = maxNumCompThreads;
maxNumCompThreads(1);
threadsCleanup = onCleanup(@() maxNumCompThreads(oldThreads)); %#ok<NASGU>

% Existing worker pools can influence serial performance. Shut them down.
oldPool = gcp('nocreate');
if ~isempty(oldPool)
    delete(oldPool);
end
poolCleanup = onCleanup(@cleanupPoolIfOpen); %#ok<NASGU>

%% ========================== SEMIDISCRETIZATION =========================
h = L/J;
x = (1:J-1)'*h;
[Xint,Yint] = meshgrid(x,x);
u0 = exp(-alpha*((Xint-L/2).^2+(Yint-L/2).^2));
u0 = u0(:);
ndof = numel(u0);
Ah = buildDiffusionMatrix2D(J,L,D);
Ih = speye(ndof);

DeltaT = T/N;
deltaT = DeltaT/M;
totalFineSteps = N*M;
rho = 8*D/h^2*cos(pi/(2*J))^2;

assert(deltaT*rho<2.785,'Fine RK4 stability check failed.');
assert((T/(Nref*MrefCoarse))*rho<2.785, ...
       'Reference coarse RK4 stability check failed.');
assert((T/(Nref*MrefFine))*rho<2.785, ...
       'Reference fine RK4 stability check failed.');

fprintf('\n============================================================\n');
fprintf('MERGED 2D FISHER-KPP: FIXED-N STRONG SCALING + TOLERANCES\n');
fprintf('J=%d, ndof=%d, T=%g, D=%g, r=%g, N=%d, M=%d\n', ...
    J,ndof,T,D,r,N,M);
fprintf('P list: %s\n',mat2str(Plist));
fprintf('K list: %s, repeats per (P,K) = %d\n',mat2str(Klist),nRepeats);
fprintf('Tolerance list: %s\n',mat2str(TolList));
fprintf('Fine RK4 stability number = %.6g\n',deltaT*rho);
fprintf('P=6 works with N=128; theory uses ceil(N/P)=%d.\n',ceil(N/6));
fprintf('All wall-clock times: mean +/- SAMPLE SD.\n');
fprintf('External LU, pool startup, references and warmups EXCLUDED.\n');
fprintf('============================================================\n');

%% ================== INDEPENDENT COMMON REFERENCE =======================
fprintf('\nCalculating independent RK4 references (NOT timed)...\n');
uRefCoarse = serialFineRK4FinalFisher(u0,Ah,r,T,Nref,MrefCoarse);
uReference = serialFineRK4FinalFisher(u0,Ah,r,T,Nref,MrefFine);
selfDifference = gridNorm2D(uRefCoarse-uReference,h);
fprintf('Reference self-difference (grid L2) = %.12e\n',selfDifference);
if ~isfinite(selfDifference) || selfDifference>min(TolList)/100
    error('Reference verification failed. Refine both reference RK4 runs.');
end

%% =========================== FIXED BE FACTOR ===========================
ttFactor = tic;
BEFixedSolver = decomposition(Ih-DeltaT*Ah,'lu');
LUSetupTime = toc(ttFactor);
fprintf('Cached BE LU factorization = %.6f s (excluded from timing).\n', ...
    LUSetupTime);

%% ====================== UNIFIED SERIAL BASELINE ========================
% Measure the complete matched fine RK4 trajectory, and also measure the
% complete N-interval sequential BE initial sweep with the same outer timer.
fprintf('\nBenchmarking matched serial RK4 and sequential coarse BE sweep...\n');
uFineWarm = serialFineRK4FinalFisher(u0,Ah,r,T,N,M); %#ok<NASGU>
uCoarseWarm = sequentialCoarseSweepFisher( ...
    u0,Ah,r,T,N,BEFixedSolver,fpTol,fpMaxit,usePredictor); %#ok<NASGU>

matchedSerialTimes = zeros(nRepeats,1);
coarseSweepTimes = zeros(nRepeats,1);
for rep=1:nRepeats
    tFine=tic;
    uMatchedFine = serialFineRK4FinalFisher(u0,Ah,r,T,N,M);
    matchedSerialTimes(rep)=toc(tFine);

    tCoarse=tic;
    uCoarse = sequentialCoarseSweepFisher( ...
        u0,Ah,r,T,N,BEFixedSolver,fpTol,fpMaxit,usePredictor); %#ok<NASGU>
    coarseSweepTimes(rep)=toc(tCoarse);

    fprintf('  Baseline run %d/%d: matched serial RK4=%.6f s, BE sweep=%.6f s\n', ...
        rep,nRepeats,matchedSerialTimes(rep),coarseSweepTimes(rep));
end

matchedSerialMean=mean(matchedSerialTimes);
matchedSerialSD=std(matchedSerialTimes);
coarseSweepMean=mean(coarseSweepTimes);
coarseSweepSD=std(coarseSweepTimes);
CF=matchedSerialMean/N;
CG=coarseSweepMean/N;
matchedReferenceError=gridNorm2D(uMatchedFine-uReference,h);
matchedMeetsAll = matchedReferenceError<=min(TolList);

fprintf('Matched serial RK4 = %.6f +/- %.6f s\n', ...
    matchedSerialMean,matchedSerialSD);
fprintf('Coarse BE sweep    = %.6f +/- %.6f s\n', ...
    coarseSweepMean,coarseSweepSD);
fprintf('Measured CF=%.9f s/slice, CG=%.9f s/slice\n',CF,CG);
fprintf('Matched RK4 vs reference: %.10e\n',matchedReferenceError);
if ~matchedMeetsAll
    warning(['Matched serial RK4 does NOT satisfy the strictest tolerance. ', ...
             'Classical speedups will not be accuracy-qualified there.']);
end

BaseTimingTable = table((1:nRepeats)',matchedSerialTimes,coarseSweepTimes, ...
    'VariableNames',{'Repeat','MatchedSerialRK4_s','SequentialCoarseBE_s'});

%% ================= SERIAL RK4: TOLERANCE-SELECTED BASELINE ===============
% A separate baseline for each tolerance, distinct from matched serial F.
% If no serial candidate qualifies, corresponding speedups remain NaN.
fprintf('\nScreening accuracy-qualified SERIAL RK4 candidates...\n');
screenNames = {'Nserial','Mserial','TotalSteps','dt','StabilityNumber', ...
               'ErrorToReference','Tested','MeetsStrictestTol'};
screenMat = nan(0,numel(screenNames));
serialStepCandidate = [];
serialCandidateError = [];

for jM=1:numel(MserialCandidates)
    Ms=MserialCandidates(jM);
    ns=Nserial*Ms;
    dt=T/ns;
    stab=dt*rho;
    qualifiedStability = (stab<2.785);

    if qualifiedStability
        yCandidate=serialFineRK4FinalFisher(u0,Ah,r,T,Nserial,Ms);
        E=gridNorm2D(yCandidate-uReference,h);
        if ~isfinite(E), E=Inf; end
    else
        E=Inf;
    end

    screenMat(end+1,:)=[Nserial,Ms,ns,dt,stab,E, ...
                         double(qualifiedStability),double(E<=min(TolList))]; %#ok<AGROW>
    serialStepCandidate(end+1)=ns; %#ok<AGROW>
    serialCandidateError(end+1)=E; %#ok<AGROW>

    fprintf('  NSteps=%7d, dt=%9.3e, stable=%d, Eref=%.7e\n', ...
        ns,dt,qualifiedStability,E);

    if E<=min(TolList)
        break;
    end
end

SerialScreeningTable=array2table(screenMat,'VariableNames',screenNames);
selectedNSteps=nan(numel(TolList),1);
for iTol=1:numel(TolList)
    idx=find(serialCandidateError<=TolList(iTol),1,'first');
    if ~isempty(idx)
        selectedNSteps(iTol)=serialStepCandidate(idx);
    else
        warning('No serial RK4 candidate satisfies tau=%.1e.',TolList(iTol));
    end
end

% Run five independent timings for each distinct qualifying serial grid.
distinctSerialSteps=unique(selectedNSteps(isfinite(selectedNSteps)),'stable');
serialSelectedTiming = nan(numel(distinctSerialSteps),3); % [steps, mean, SD]
for iS=1:numel(distinctSerialSteps)
    ns=distinctSerialSteps(iS);
    fprintf('  Timed accuracy-selected serial RK4, total steps=%d\n',ns);
    warm=serialFineRK4FinalFisher(u0,Ah,r,T,1,ns); %#ok<NASGU>
    tvals=zeros(nRepeats,1);
    for rep=1:nRepeats
        tt=tic;
        ySel=serialFineRK4FinalFisher(u0,Ah,r,T,1,ns);
        tvals(rep)=toc(tt);
    end
    serialSelectedTiming(iS,:)=[ns,mean(tvals),std(tvals)];
    fprintf('    %.6f +/- %.6f s, Eref=%.8e\n', ...
        mean(tvals),std(tvals),gridNorm2D(ySel-uReference,h));
end

accuracySerialMean=nan(numel(TolList),1);
accuracySerialSD=nan(numel(TolList),1);
accuracySerialError=nan(numel(TolList),1);
for iTol=1:numel(TolList)
    ns=selectedNSteps(iTol);
    if ~isfinite(ns), continue; end
    iS=find(serialSelectedTiming(:,1)==ns,1);
    iC=find(serialStepCandidate==ns,1);
    accuracySerialMean(iTol)=serialSelectedTiming(iS,2);
    accuracySerialSD(iTol)=serialSelectedTiming(iS,3);
    accuracySerialError(iTol)=serialCandidateError(iC);
end
SerialSelectedTable = table(TolList(:),selectedNSteps,accuracySerialError, ...
    accuracySerialMean,accuracySerialSD, ...
    'VariableNames',{'Tolerance','SelectedSerialRK4Steps', ...
                     'FinalErrorRef','TimeMean_s','TimeSD_s'});

fprintf('\nSerial RK4 accuracy-qualified table:\n');
disp(SerialSelectedTable);

%% ============================ PARALLEL RUNS =============================
% Values from PKTable are not repeated by tau. A full tolerance-labeled
% output is constructed after all (P,K) measurements finish.
pkNames = { ...
    'P','N','M','K','FinalErrorRefMean','FinalErrorRefMax', ...
    'FinalErrorToMatchedMean','MatchedSerialMean_s','MatchedSerialSD_s', ...
    'Tpred_s','TactualMean_s','TactualSD_s', ...
    'SpeedupPredMatched','SpeedupActualMatched', ...
    'EfficiencyPredMatched_pct','EfficiencyActualMatched_pct', ...
    'TaskIdealMean_s','TaskIdealSD_s','TaskIdealSpeedupMatched', ...
    'InitialCoarseMean_s','InitialCoarseSD_s', ...
    'ParallelFineMean_s','ParallelFineSD_s', ...
    'CorrectionMean_s','CorrectionSD_s','OtherMean_s','OtherSD_s', ...
    'MeanFPIterations','MaxFPIterations','MaxBEResidual','BE_LUSetup_s', ...
    'CF_s_perSlice','CG_s_perSlice'};
PKMatrix=nan(0,numel(pkNames));

rawNames = {'P','K','Repeat','ErrorToReference','ErrorToMatched', ...
    'Tactual_s','InitialCoarse_s','ParallelFine_s','Correction_s','Other_s', ...
    'TaskIdeal_s','TaskIdealCoarse_s','TaskIdealFineLowerBound_s', ...
    'MeanFPIterations','MaxFPIterations','MaxBEResidual'};
RawMatrix=nan(0,numel(rawNames));

phaseNames={'P','TotalK','Repeat','Iteration','FineWall_s','CorrectionWall_s'};
PhaseMatrix=nan(0,numel(phaseNames));

for iP=1:numel(Plist)
    P=Plist(iP);
    fprintf('\n============================================================\n');
    fprintf('P=%d of N=%d (fine-slice batches: ceil(N/P)=%d)\n', ...
        P,N,ceil(N/P));
    fprintf('============================================================\n');

    existingPool=gcp('nocreate');
    if ~isempty(existingPool)
        delete(existingPool);
    end
    pool=parpool('local',P);
    assert(pool.NumWorkers==P,'Requested P=%d but acquired a different pool.',P);

    % Per-P warmups, excluded. Keep the same BE factorization for all P.
    workerWarm=zeros(P,1);
    parfor iw=1:P
        vWarm=finePropRK4Fisher(u0,Ah,r,DeltaT,M);
        workerWarm(iw)=vWarm(1);
    end
    assert(all(isfinite(workerWarm)),'Worker RK4 warm-up failed.');

    Nwarm=min(N,P);
    Twarm=Nwarm*DeltaT;
    [~,warmTiming]=runPararealBE_RK4_ParallelFisherFixedPoint( ...
        u0,Ah,r,BEFixedSolver,Twarm,Nwarm,M,1, ...
        fpTol,fpMaxit,usePredictor);
    assert(warmTiming.failedCoarseSolves==0,'Parareal warmup failed.');

    for K=Klist
        Tactual=zeros(nRepeats,1);
        Tinit=zeros(nRepeats,1);
        Tfine=zeros(nRepeats,1);
        Tcorr=zeros(nRepeats,1);
        Tother=zeros(nRepeats,1);
        Tideal=zeros(nRepeats,1);
        TidealCoarse=zeros(nRepeats,1);
        TidealFine=zeros(nRepeats,1);
        Eref=zeros(nRepeats,1);
        Ematched=zeros(nRepeats,1);
        meanFP=zeros(nRepeats,1);
        maxFP=zeros(nRepeats,1);
        maxResidual=zeros(nRepeats,1);

        % Time each K independently, as in the original two scripts.
        for rep=1:nRepeats
            tt=tic;
            [Upar,timing]=runPararealBE_RK4_ParallelFisherFixedPoint( ...
                u0,Ah,r,BEFixedSolver,T,N,M,K, ...
                fpTol,fpMaxit,usePredictor);
            Tactual(rep)=toc(tt);

            assert(timing.failedCoarseSolves==0, ...
                'BE coarse solver failed at P=%d,K=%d,rep=%d.',P,K,rep);
            uEnd=Upar(end,:).';
            Eref(rep)=gridNorm2D(uEnd-uReference,h);
            Ematched(rep)=gridNorm2D(uEnd-uMatchedFine,h);
            assert(isfinite(Eref(rep)),'Nonfinite Parareal error.');

            Tinit(rep)=timing.initialCoarse;
            Tfine(rep)=sum(timing.fine);
            Tcorr(rep)=sum(timing.correction);
            Tother(rep)=Tactual(rep)-Tinit(rep)-Tfine(rep)-Tcorr(rep);

            % Task-level "ideal no-dispatch" estimate based on original
            % StrongScaling instrumentation. With N/P noninteger, do not
            % treat max interval task duration as the complete fine phase.
            TidealCoarse(rep)=sum(timing.coarseInitialPerInterval) ...
                +sum(timing.coarseCorrectionPerInterval(:)) ...
                +sum(timing.correctionAlgebraPerInterval(:));
            idealFine=0;
            for kk=1:K
                tasks=timing.finePerInterval(kk,:);
                idealFine=idealFine + max(max(tasks),sum(tasks)/P);
                PhaseMatrix(end+1,:)=[P,K,rep,kk, ...
                    timing.fine(kk),timing.correction(kk)]; %#ok<AGROW>
            end
            TidealFine(rep)=idealFine;
            Tideal(rep)=TidealCoarse(rep)+TidealFine(rep);

            meanFP(rep)=timing.meanFPIterations;
            maxFP(rep)=timing.maxFPIterations;
            maxResidual(rep)=timing.maxBEResidual;

            RawMatrix(end+1,:)=[P,K,rep,Eref(rep),Ematched(rep), ...
                Tactual(rep),Tinit(rep),Tfine(rep),Tcorr(rep),Tother(rep), ...
                Tideal(rep),TidealCoarse(rep),TidealFine(rep), ...
                meanFP(rep),maxFP(rep),maxResidual(rep)]; %#ok<AGROW>

            fprintf('  P=%2d K=%d repeat %d/%d: real=%.6fs, Eref=%.5e\n', ...
                P,K,rep,nRepeats,Tactual(rep),Eref(rep));
        end

        % Analytical cost model from serial kernel benchmarks.
        Tpred=(K+1)*N*CG + K*ceil(N/P)*CF;
        predSpeedupMatched=matchedSerialMean/Tpred;
        actualSpeedupMatched=matchedSerialMean/mean(Tactual);
        predEffMatched=100*predSpeedupMatched/P;
        actualEffMatched=100*actualSpeedupMatched/P;

        pkRow=[P,N,M,K,mean(Eref),max(Eref),mean(Ematched), ...
            matchedSerialMean,matchedSerialSD, ...
            Tpred,mean(Tactual),std(Tactual), ...
            predSpeedupMatched,actualSpeedupMatched, ...
            predEffMatched,actualEffMatched, ...
            mean(Tideal),std(Tideal),matchedSerialMean/mean(Tideal), ...
            mean(Tinit),std(Tinit),mean(Tfine),std(Tfine), ...
            mean(Tcorr),std(Tcorr),mean(Tother),std(Tother), ...
            mean(meanFP),max(maxFP),max(maxResidual),LUSetupTime,CF,CG];

        assert(numel(pkRow)==numel(pkNames),'PK table width mismatch.');
        PKMatrix(end+1,:)=pkRow; %#ok<AGROW>

        fprintf(['  SUMMARY P=%d K=%d: Tpred=%.6fs, Treal=%.6f +/- %.6fs, ', ...
            'Spred=%.4f, Sreal=%.4f, Effreal=%.2f%%, Eref(max)=%.4e\n'], ...
            P,K,Tpred,mean(Tactual),std(Tactual), ...
            predSpeedupMatched,actualSpeedupMatched,actualEffMatched,max(Eref));
    end

    delete(pool);

    % Save a recoverable numerical checkpoint after every completed P.
    % This does NOT include open parallel resources or large solution arrays.
    if saveResults
        save('FisherKPP2D_Merged_Checkpoint.mat', ...
            'PKMatrix','RawMatrix','PhaseMatrix', ...
            'pkNames','rawNames','phaseNames','Plist','Klist','TolList', ...
            'matchedSerialMean','matchedSerialSD', ...
            'matchedReferenceError','CF','CG');
    end
end

PKTable=array2table(PKMatrix,'VariableNames',pkNames);
RawTimingTable=array2table(RawMatrix,'VariableNames',rawNames);
PerIterationStagesTable=array2table(PhaseMatrix,'VariableNames',phaseNames);

%% ===================== LABEL EACH ERROR TOLERANCE ======================
% Same (P,K) timing is reused under each tau, NOT measured again.
% Both speedup denominators are identified explicitly in the columns.
FullByToleranceTable=table();
for iTol=1:numel(TolList)
    tau=TolList(iTol);
    block=PKTable;

    block.Tolerance=repmat(tau,height(block),1);
    block.Qualified=(block.FinalErrorRefMax<=tau) & ...
                    (matchedReferenceError<=tau);
    block.AccuracySerialMean_s=repmat(accuracySerialMean(iTol),height(block),1);
    block.AccuracySerialSD_s=repmat(accuracySerialSD(iTol),height(block),1);
    block.AccuracySerialSteps=repmat(selectedNSteps(iTol),height(block),1);

    block.SpeedupPredAccuracy=block.AccuracySerialMean_s./block.Tpred_s;
    block.SpeedupActualAccuracy=block.AccuracySerialMean_s./block.TactualMean_s;
    block.EfficiencyPredAccuracy_pct=100*block.SpeedupPredAccuracy./block.P;
    block.EfficiencyActualAccuracy_pct=100*block.SpeedupActualAccuracy./block.P;

    % A speedup value is "accuracy-qualified" only if Parareal AND the
    % relevant serial baseline satisfy the SAME tau.
    block.QualifiedSpeedupPredMatched=block.SpeedupPredMatched;
    block.QualifiedSpeedupActualMatched=block.SpeedupActualMatched;
    block.QualifiedSpeedupPredAccuracy=block.SpeedupPredAccuracy;
    block.QualifiedSpeedupActualAccuracy=block.SpeedupActualAccuracy;

    block.QualifiedSpeedupPredMatched(~block.Qualified)=NaN;
    block.QualifiedSpeedupActualMatched(~block.Qualified)=NaN;
    qAccuracy=block.Qualified & isfinite(block.AccuracySerialMean_s);
    block.QualifiedSpeedupPredAccuracy(~qAccuracy)=NaN;
    block.QualifiedSpeedupActualAccuracy(~qAccuracy)=NaN;

    block=movevars(block,{'Tolerance','Qualified'},'Before',1);
    if iTol==1
        FullByToleranceTable=block;
    else
        FullByToleranceTable=[FullByToleranceTable;block]; %#ok<AGROW>
    end
end

%% ================== FIRST QUALIFYING K BY (tau,P) =======================
% K<=3. If no tested K satisfies tau, SelectedK stays NaN.
selectedNames={'Tolerance','P','SelectedK','Qualified', ...
    'FinalErrorRefMax','Tpred_s','TactualMean_s','TactualSD_s', ...
    'SpeedupPredMatched','SpeedupActualMatched', ...
    'EfficiencyPredMatched_pct','EfficiencyActualMatched_pct', ...
    'AccuracySerialMean_s','AccuracySerialSD_s', ...
    'SpeedupPredAccuracy','SpeedupActualAccuracy', ...
    'EfficiencyPredAccuracy_pct','EfficiencyActualAccuracy_pct', ...
    'TaskIdealMean_s','TaskIdealSD_s'};
SelectedMatrix=nan(numel(TolList)*numel(Plist),numel(selectedNames));

iRow=0;
for iTol=1:numel(TolList)
    for iP=1:numel(Plist)
        iRow=iRow+1;
        tau=TolList(iTol);
        P=Plist(iP);
        SelectedMatrix(iRow,1:2)=[tau,P];
        SelectedMatrix(iRow,4)=0; % not qualified unless a K is found
        eligible=find(FullByToleranceTable.Tolerance==tau & ...
                      FullByToleranceTable.P==P & ...
                      FullByToleranceTable.Qualified);
        if isempty(eligible)
            fprintf('NO qualifying K<=%d: tau=%.1e, P=%d\n',Kmax,tau,P);
            continue;
        end

        [~,firstKPos]=min(FullByToleranceTable.K(eligible));
        idx=eligible(firstKPos);
        rec=FullByToleranceTable(idx,:);

        row=[tau,P,rec.K,1,rec.FinalErrorRefMax, ...
            rec.Tpred_s,rec.TactualMean_s,rec.TactualSD_s, ...
            rec.SpeedupPredMatched,rec.SpeedupActualMatched, ...
            rec.EfficiencyPredMatched_pct,rec.EfficiencyActualMatched_pct, ...
            rec.AccuracySerialMean_s,rec.AccuracySerialSD_s, ...
            rec.SpeedupPredAccuracy,rec.SpeedupActualAccuracy, ...
            rec.EfficiencyPredAccuracy_pct,rec.EfficiencyActualAccuracy_pct, ...
            rec.TaskIdealMean_s,rec.TaskIdealSD_s];
        SelectedMatrix(iRow,:)=row;
    end
end
SelectedByToleranceTable=array2table(SelectedMatrix, ...
    'VariableNames',selectedNames);

%% ============================= PRESENT ================================
fprintf('\n============================================================\n');
fprintf('COMPLETE PER-(P,K) MODEL AND MEASUREMENTS (15 rows)\n');
fprintf('THESE TIMES ARE INDEPENDENT OF THE REQUESTED ERROR TOLERANCE.\n');
fprintf('============================================================\n');
disp(PKTable(:,{'P','K','FinalErrorRefMax','Tpred_s', ...
    'TactualMean_s','TactualSD_s','SpeedupPredMatched', ...
    'SpeedupActualMatched','EfficiencyPredMatched_pct', ...
    'EfficiencyActualMatched_pct','TaskIdealMean_s'}));

fprintf('\n============================================================\n');
fprintf('FIRST QUALIFYING K FOR EVERY (TOLERANCE,P) (25 rows)\n');
fprintf('NaN SelectedK means none of K=1,2,3 satisfies tau.\n');
fprintf('Matched and accuracy-selected speedups use DIFFERENT serial baselines.\n');
fprintf('============================================================\n');
disp(SelectedByToleranceTable);

%% =============================== SAVE ================================
if saveResults
    writetable(PKTable,'FisherKPP2D_Merged_PerPK.csv');
    writetable(FullByToleranceTable,'FisherKPP2D_Merged_AllTolerances.csv');
    writetable(SelectedByToleranceTable,'FisherKPP2D_Merged_SelectedByTolerance.csv');
    writetable(RawTimingTable,'FisherKPP2D_Merged_RawPararealTimings.csv');
    writetable(PerIterationStagesTable, ...
        'FisherKPP2D_Merged_PerIterationStages.csv');
    writetable(SerialScreeningTable, ...
        'FisherKPP2D_Merged_SerialAccuracyScreening.csv');
    writetable(SerialSelectedTable, ...
        'FisherKPP2D_Merged_SerialAccuracySelected.csv');
    writetable(BaseTimingTable, ...
        'FisherKPP2D_Merged_BaselineRawTimings.csv');

    save('FisherKPP2D_Merged_Results.mat', ...
        'PKTable','FullByToleranceTable','SelectedByToleranceTable', ...
        'RawTimingTable','PerIterationStagesTable', ...
        'SerialScreeningTable','SerialSelectedTable', ...
        'BaseTimingTable','matchedSerialMean','matchedSerialSD', ...
        'matchedReferenceError','CF','CG','Plist','TolList','Klist', ...
        'J','N','M','T','LUSetupTime','selfDifference');
end

if makePlots
    plotScalingVsP(PKTable,Plist,Klist, ...
        'SpeedupPredMatched','Predicted matched-RK4 speedup', ...
        'FisherKPP2D_Merged_PredictedSpeedup.png');
    plotScalingVsP(PKTable,Plist,Klist, ...
        'SpeedupActualMatched','Measured matched-RK4 speedup', ...
        'FisherKPP2D_Merged_MeasuredSpeedup.png');
end

fprintf('\n============================================================\n');
fprintf('DONE: 8 CSV files + MAT, and optionally 2 PNG figures.\n');
fprintf('Key paper table: FisherKPP2D_Merged_SelectedByTolerance.csv\n');
fprintf('Full detail: FisherKPP2D_Merged_AllTolerances.csv\n');
fprintf('Theoretical Tpred uses ceil(N/P) and benchmarked CF, CG.\n');
fprintf('TaskIdealMean is an optimistic no-dispatch proxy, not actual time.\n');
fprintf('Wallclock model gap is NOT measured communication time.\n');
fprintf('============================================================\n');

end

%% ====================== ADDITIONAL LOCAL FUNCTIONS =====================
function y = sequentialCoarseSweepFisher( ...
    u0,A,r,T,N,BEFixedSolver,fpTol,fpMaxit,usePredictor)
% Same N-slice BE solver as used in Parareal initial coarse prediction.
    dt=T/N;
    y=u0(:);
    for nn=1:N
        [y,info]=coarseBackwardEulerFisherFixedPoint( ...
            y,A,r,dt,BEFixedSolver,fpTol,fpMaxit,usePredictor);
        if ~info.converged
            error('Coarse baseline BE did not converge at interval %d.',nn);
        end
    end
end

function plotScalingVsP(PKTable,Plist,Klist,columnName,yLabel,outFile)
% Make a plain speedup curve, without any dependence on annotation/UI state.
    try
        f=figure('Visible','off','Color','w');
        ax=axes('Parent',f);
        hold(ax,'on');
        handleVec=gobjects(numel(Klist),1);
        labels=cell(numel(Klist),1);
        for ik=1:numel(Klist)
            K=Klist(ik);
            values=nan(numel(Plist),1);
            for ip=1:numel(Plist)
                idx=find(PKTable.P==Plist(ip) & PKTable.K==K,1);
                if ~isempty(idx)
                    colValues=PKTable.(columnName);
                    values(ip)=colValues(idx);
                end
            end
            handleVec(ik)=plot(ax,Plist,values,'-o', ...
                'LineWidth',1.3,'MarkerSize',5);
            labels{ik}=sprintf('K=%d',K);
        end
        xlabel(ax,'Number of workers P');
        ylabel(ax,yLabel);
        title(ax,'Fisher--KPP: J=160, N=128, M=500');
        set(ax,'XTick',Plist);
        grid(ax,'on');
        legend(ax,handleVec,labels,'Location','best');
        saveas(f,outFile);
        close(f);
    catch ME
        warning('Could not save plot %s: %s',outFile,ME.message);
    end
end

function cleanupPoolIfOpen()
    try
        pool=gcp('nocreate');
        if ~isempty(pool)
            delete(pool);
        end
    catch
        % Preserve original exception, if any.
    end
end

function A = buildDiffusionMatrix2D(J,L,D)
    h = L/J;
    n = J-1;
    e = ones(n,1);
    T1 = spdiags([e -2.0*e e],[-1 0 1],n,n);
    I1 = speye(n);
    Lap2D = kron(I1,T1) + kron(T1,I1);
    A = (D/h^2) * Lap2D;
end

% -------------------------------------------------------------------------

function y = serialFineRK4FinalFisher(u0,A,r,T,N,Mfine)
    DeltaT = T/N;
    y = u0(:);
    for n = 1:N
        y = finePropRK4Fisher(y,A,r,DeltaT,Mfine);
    end
end

% -------------------------------------------------------------------------

function [Upar,timing] = runPararealBE_RK4_ParallelFisherFixedPoint( ...
    u0,A,r,BEFixedSolver,T,N,Mfine,K,fpTol,fpMaxit,usePredictor)

    ndof = numel(u0);
    DeltaT = T/N;

    timing.initialCoarse = 0.0;
    timing.fine          = zeros(K,1);
    timing.correction    = zeros(K,1);
    timing.coarseInitialPerInterval = zeros(1,N);
    timing.coarseCorrectionPerInterval = zeros(K,N);
    timing.correctionAlgebraPerInterval = zeros(K,N);
    timing.finePerInterval = zeros(K,N);

    allIterations = [];
    maxResidual   = 0.0;
    failedSolves  = 0;

    % ---- initial coarse prediction ----
    timerInitial = tic;
    Uold = zeros(N+1,ndof);
    Uold(1,:) = u0(:).';
    GoldStored = zeros(N,ndof);

    for n = 1:N
        v = Uold(n,:).';
        tG = tic;
        [Gv,info] = coarseBackwardEulerFisherFixedPoint( ...
            v,A,r,DeltaT,BEFixedSolver,fpTol,fpMaxit,usePredictor);
        timing.coarseInitialPerInterval(n) = toc(tG);
        allIterations(end+1,1) = info.iterations;
        maxResidual = max(maxResidual,info.residual);
        if ~info.converged
            failedSolves = failedSolves + 1;
        end
        Uold(n+1,:)     = Gv.';
        GoldStored(n,:) = Gv.';
    end
    timing.initialCoarse = toc(timerInitial);

    % ---- Parareal iterations ----
    for k = 1:K
        fineOld = zeros(N,ndof);
        fineTaskSeconds = zeros(N,1);  % sliced output from parfor

        timerFine = tic;
        parfor n = 1:N
            vOld = Uold(n,:).';
            taskClock = tic;
            Fv = finePropRK4Fisher(vOld,A,r,DeltaT,Mfine);
            fineTaskSeconds(n) = toc(taskClock);
            fineOld(n,:) = Fv.';
        end
        timing.fine(k) = toc(timerFine);
        timing.finePerInterval(k,:) = fineTaskSeconds(:).';

        if any(~isfinite(fineOld(:)))
            error('Parallel fine RK4 produced NaN or Inf.');
        end

        timerCorrection = tic;
        Unew = zeros(N+1,ndof);
        Unew(1,:) = u0(:).';
        GnewStored = zeros(N,ndof);

        for n = 1:N
            vNew = Unew(n,:).';
            tG = tic;
            [Gnew,info] = coarseBackwardEulerFisherFixedPoint( ...
                vNew,A,r,DeltaT,BEFixedSolver,fpTol,fpMaxit,usePredictor);
            timing.coarseCorrectionPerInterval(k,n) = toc(tG);
            allIterations(end+1,1) = info.iterations;
            maxResidual = max(maxResidual,info.residual);
            if ~info.converged
                failedSolves = failedSolves + 1;
            end
            tVec = tic;
            Unew(n+1,:) = (Gnew + fineOld(n,:).' - GoldStored(n,:).').';
            if any(~isfinite(Unew(n+1,:)))
                error('Parareal correction produced NaN or Inf.');
            end
            GnewStored(n,:) = Gnew.';
            timing.correctionAlgebraPerInterval(k,n) = toc(tVec);
        end
        timing.correction(k) = toc(timerCorrection);

        Uold        = Unew;
        GoldStored  = GnewStored;
    end

    Upar = Uold;

    if isempty(allIterations)
        timing.meanFPIterations = 0;
        timing.maxFPIterations  = 0;
    else
        timing.meanFPIterations = mean(allIterations);
        timing.maxFPIterations  = max(allIterations);
    end
    timing.maxBEResidual     = maxResidual;
    timing.failedCoarseSolves = failedSolves;
end

% -------------------------------------------------------------------------

function [y,info] = coarseBackwardEulerFisherFixedPoint( ...
    v,A,r,DeltaT,BEFixedSolver,tol,maxit,usePredictor)

    v = v(:);
    if any(~isfinite(v))
        error('Backward Euler received NaN or Inf.');
    end

    if usePredictor
        y = v + DeltaT*(A*v + r*v.*(1-v));
    else
        y = v;
    end

    if any(~isfinite(y))
        y = v;
    end

    R        = y - v - DeltaT*(A*y + r*y.*(1-y));
    residual = norm(R,inf);

    if residual <= tol
        info.converged  = true;
        info.iterations = 0;
        info.residual   = residual;
        return;
    end

    converged = false;
    for iter = 1:maxit
        rhs  = v + DeltaT*r*y.*(1-y);
        ynew = BEFixedSolver\rhs;

        if any(~isfinite(ynew))
            error('BE fixed-point iteration produced NaN or Inf.');
        end

        Rnew     = ynew - v - DeltaT*(A*ynew + r*ynew.*(1-ynew));
        residual = norm(Rnew,inf);
        y        = ynew;

        if residual <= tol
            converged = true;
            break;
        end
    end

    info.converged  = converged;
    info.iterations = iter;
    info.residual   = residual;
end

% -------------------------------------------------------------------------

function y = finePropRK4Fisher(v,A,r,DeltaT,M)
    dt = DeltaT/M;
    y  = v(:);
    for m = 1:M
        y = rk4StepFisher(y,A,r,dt);
        if any(~isfinite(y))
            error('Fine RK4 produced NaN or Inf at step %d of %d.',m,M);
        end
    end
end

% -------------------------------------------------------------------------

function ynew = rk4StepFisher(y,A,r,dt)
    k1 = A*y        + r*y.*(1-y);
    y2 = y + 0.5*dt*k1;
    k2 = A*y2       + r*y2.*(1-y2);
    y3 = y + 0.5*dt*k2;
    k3 = A*y3       + r*y3.*(1-y3);
    y4 = y + dt*k3;
    k4 = A*y4       + r*y4.*(1-y4);
    ynew = y + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
end

% -------------------------------------------------------------------------

function nr = gridNorm2D(u,h)
    nr = sqrt(h^2 * sum(abs(u).^2));
end
