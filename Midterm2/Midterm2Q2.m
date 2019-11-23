%% EECE5644 Midterm Exam 2 
% Question 1
% Benjamin Gincley
% 16 November 2019
%% Initialize the problem
% Load data
dataTrain = readtable('Q2train.csv');
timeTrain = dataTrain.Var1;
hTrain = dataTrain.Var2;
bTrain = dataTrain.Var3;
nTrain = size(timeTrain,1);
dataTest = readtable('Q2test.csv');
timeTest = dataTest.Var1;
hTest = dataTest.Var2;
bTest = dataTest.Var3;
nTest = size(timeTest,1);

% Plot points in coordinate system
fig1 = figure(); hold on
plot(hTrain,bTrain,'--.k')
title('Noisy trajectory of training data')
xlabel('longitude position')
ylabel('latitiude position')

%% Initialize model
clear state
% Assign initial conditions
t = 2; % initial time
state.C = [1 0 0]; % sensor measures position only
state.A = [1 t 0.5*t^2; 0 1 t; 0 0 1]; % motion equation governing state dynamics
state.Xh = [0; 0; 0]; % initial longitude state in model
state.Xb = [0; 0; 0]; % initial latitude state in model
state.P = 0.5;    % initial P value (chosen arbitrarily)
state.Q = 0.1*eye(3);  % noise of the model (pre-optimized value)
state.R = 1*eye(1);    % noise of the measurement (pre-optimized value)
% Implementing the Kalman Filter
% Loop through all training points 
for i=1:nTrain
    % Assign training measurements in state model
    state(end).time = timeTrain(i);
    state(end).Zh = hTrain(i);
    state(end).Zb = bTrain(i);
    if i ~= max(nTrain) % Avoids creating nTrain+1 points
        state(end+1) = kalmanfilterfxn(state(end)); % Iterate the kalman filter function
    end
end
for i=1:nTrain % extracts values from state struct for plotting
    xh(i) = state(i).Xh(1);
    xb(i) = state(i).Xb(1);
    stateTime(i) = state(i).time;
end
plot(xh,xb,'-xr')
%% Interpolate from model output
% Interpolate from the kalman filter output
[timeModelInterp, hModelInterp, bModelInterp] = interpolate(xh,xb,stateTime,2,2);
% The following figures demonstrate the output behavior of the
% interpolation function
figure(); hold on   % longitude vs time
plot(stateTime,xh,'-k.')
plot(timeModelInterp,hModelInterp,'xb')
title('Interpolated Longitude Coordinate vs Time from Kalman Filter Model')
xlabel('Time')
ylabel('Longitude')
legend('Model Output','Interpolated Value','location','southeast')
figure(); hold on   % coordinate plot
plot(xh,xb,'-k.')
plot(hModelInterp,bModelInterp,'xb')
title('Interpolated Longitude and Latitude from Kalman Filter Model')
xlabel('Longitude')
ylabel('Latitude')
legend('Model Output','Interpolated Value','location','southeast')
%% Tune Hyperparameters K and S
% Initialize test set for K and S and search through logspace
K_space = logspace(-4,4,64);
S_space = logspace(-4,4,64);
nK = size(K_space,2);
nS = size(S_space,2);
error = zeros(nK,nS);
[K_Mesh, S_Mesh] = meshgrid(K_space,S_space);
% Iteratively run the kalman filter model for each combination of K and S
% Store the error in a matrix for minimum finding
for i=1:nK
    for j=1:nS
        clear state
        K = K_space(i);
        S = S_space(j);
        [state,xh,xb,stateTime,nPoints] = iteratekalmanfilter(K,S,timeTrain,hTrain,bTrain);
        [timeModelInterp, hModelInterp, bModelInterp] = interpolate(xh,xb,stateTime,2,2);
        hError = (hModelInterp-hTest(1:end-1)').^2;
        bError = (bModelInterp-bTest(1:end-1)').^2;
        error(i,j) = sum((hError+bError)/(nPoints));
    end
end
% Plot the contour map of error in log space
figure(); hold on;
contourf(log10(K_Mesh),log10(S_Mesh),error,16);
colorbar;
title('Cross Validation Error Contour Plot');
xlabel('log(K) value');
ylabel('log(S) value');
% Calculate the minimum error, plot it, and store the values of K and S
minError = min(error(:))
[minS, minK] = find(error==minError)
minS = S_space(minS)
minK = K_space(minK)
scatter(log10(minK),log10(minS),'*r')
legend('Error contour','Local minimum error')
%% Evaluating the model at optimal parameters
% Using the minimum K and S values, run the kalman filter one more time
% Plot two versions, one using training points as measurements and one
% using testing points as measurements
% Test point version chosen for presentation
[state,xh,xb,stateTime,nPoints] = iteratekalmanfilter(minS,minK,timeTrain,hTrain,bTrain);
figure(); hold on
plot(hTrain,bTrain,'ok');
plot(hTest,bTest,'ob');
%plot(xh,xb,'--rx');
title('Kalman filter overlaid on training and test data')
xlabel('longitude position')
ylabel('latitiude position')
[state,xh,xb,stateTime,nPoints] = iteratekalmanfilter(minS,minK,timeTest,hTest,bTest);
plot(xh,xb,'--mx');
%legend('Train data','Test data','Kalman Output train data','Kalman Output test data')
legend('Train data','Test data','Kalman Output')
%% Function
function state = kalmanfilterfxn(state)
    % This function applies the Kalman filter function to state variables
    % for both h and b dimensions, and updates the state on each function
    % call
    state.Xh = state.A*state.Xh;    % Pass h state variables through A
    state.Xb = state.A*state.Xb;    % Pass b state variables through A
    state.P = state.A*state.P*state.A'+state.Q;     % Update covariance of state
    KalGain = state.P*state.C'*inv(state.C*state.P*state.C'+state.R);   % Calculate Kalman gain
    state.Xh = state.Xh+KalGain*(state.Zh-state.C*state.Xh);    % Update h state variables using Kalman gain
    state.Xb = state.Xb+KalGain*(state.Zb-state.C*state.Xb);    % Update b state variables using Kalman gain
    state.P = state.P-KalGain*state.C*state.P;  % Update covariance with Kalman gain
end

function [interpTime,hInterpPoints,bInterpPoints] = interpolate(hpoints, bpoints, timeVec, interval, timeInit)
    % This specialized function applies a linear interpolation to both h
    % and b points and returns new time values and their associated points
    interpTime = timeInit:interval:max(timeVec);    % Create new time vector to be mapped
    hInterpPoints = interp1(timeVec, hpoints, interpTime, 'linear');     % Linear interpolation over h points
    bInterpPoints = interp1(timeVec, bpoints, interpTime, 'linear');     % Linear interpolation over b points
end

function [state,xh,xb,stateTime,nPoints] = iteratekalmanfilter(K,S,timeVec,hVec,bVec)
    % This function re-initializes the state space and has the kalmanfilter
    % function embedded for iterative fitting
    nPoints = size(timeVec,1);
    t = 2; % initial time
    state.C = [1 0 0]; % initial measurements
    state.A = [1 t 0.5*t^2; 0 1 t; 0 0 1]; % motion equation governing state dynamics
    state.Xh = [0; 0; 0]; % initial longitude state in model
    state.Xb = [0; 0; 0]; % initial latitude state in model
    state.P = 0.5;    % initial P value
    state.Q = K*eye(3);  % noise of the model
    state.R = S*eye(1);    % noise of the measurement
    % Implementing the Kalman Filter
    % Loop through all training points 
    for i=1:nPoints
        % Assign training measurements in state model
        state(end).time = timeVec(i);
        state(end).Zh = hVec(i);
        state(end).Zb = bVec(i);
        if i ~= max(nPoints)% Avoids creating nTrain+1 points
            state(end+1) = kalmanfilterfxn(state(end)); % Iterate the kalman filter function
        end
    end
    for i=1:nPoints % extracts values from state struct for plotting
        xh(i) = state(i).Xh(1);
        xb(i) = state(i).Xb(1);
        stateTime(i) = state(i).time;
    end
end