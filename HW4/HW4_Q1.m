%% Homework 4
% EECE5644
% Benjamin Gincley
% 8 November 2019
%%
clear all
close all
%% Load images and initialize feature matrix
% Load image
imgBird = imread('42049_colorBird.jpg');
figure(); imshow(imgBird)
% Extract normalized RGB data
imgBirdRGB = double(imgBird)/256;
imgBirdR = imgBirdRGB(:,:,1);
imgBirdG = imgBirdRGB(:,:,2);
imgBirdB = imgBirdRGB(:,:,3);
% Set up coordinate system
maxRowPos = size(imgBirdR,1);
maxColPos = size(imgBirdR,2);
rowPos = linspace(0,1,maxRowPos);
colPos = linspace(0,1,maxColPos);
rowPosVec = repmat(rowPos',[maxColPos 1]);
for i = 1:maxColPos
    colPosVecNew = repmat(colPos(i),[maxRowPos 1]);
    ise = evalin('base', 'exist(''colPosVec'',''var'') == 1');
    if ise
        colPosVec = cat(1,colPosVec,colPosVecNew);
    else
        colPosVec = colPosVecNew;
    end
end
% Reshape from rectangle to 1D vector
imgBirdRVec = reshape(imgBirdR,maxRowPos*maxColPos,1);
imgBirdGVec = reshape(imgBirdG,maxRowPos*maxColPos,1);
imgBirdBVec = reshape(imgBirdB,maxRowPos*maxColPos,1);
% Assign features matrix
birdFeatures = zeros(154401,5);
birdFeatures(:,1) = colPosVec;
birdFeatures(:,2) = rowPosVec;
birdFeatures(:,3) = imgBirdRVec;
birdFeatures(:,4) = imgBirdGVec;
birdFeatures(:,5) = imgBirdBVec;
% Repeat with plane
imgPlane = imread('3096_colorPlane.jpg');
figure(); imshow(imgPlane)
imgPlaneRGB = double(imgPlane)/255;
imgPlaneR = imgPlaneRGB(:,:,1);
imgPlaneG = imgPlaneRGB(:,:,2);
imgPlaneB = imgPlaneRGB(:,:,3);
imgPlaneRVec = reshape(imgPlaneR,maxRowPos*maxColPos,1);
imgPlaneGVec = reshape(imgPlaneG,maxRowPos*maxColPos,1);
imgPlaneBVec = reshape(imgPlaneB,maxRowPos*maxColPos,1);
planeFeatures = zeros(154401,5);
planeFeatures(:,1) = colPosVec;
planeFeatures(:,2) = rowPosVec;
planeFeatures(:,3) = imgPlaneRVec;
planeFeatures(:,4) = imgPlaneGVec;
planeFeatures(:,5) = imgPlaneBVec;
%% Create K-Means
rng(0)
birdKM2 = kmeans(birdFeatures,2,'Distance','sqeuclidean','Display','Final');
birdKM3 = kmeans(birdFeatures,3,'Distance','sqeuclidean','Display','Final');
birdKM4 = kmeans(birdFeatures,4,'Distance','sqeuclidean','Display','Final');
birdKM5 = kmeans(birdFeatures,5,'Distance','sqeuclidean','Display','Final');

planeKM2 = kmeans(planeFeatures,2,'Distance','sqeuclidean','Display','Final');
planeKM3 = kmeans(planeFeatures,3,'Distance','sqeuclidean','Display','Final');
planeKM4 = kmeans(planeFeatures,4,'Distance','sqeuclidean','Display','Final');
planeKM5 = kmeans(planeFeatures,5,'Distance','sqeuclidean','Display','Final');
%% Decode K-Means to images
decodeLabels(birdKM2,maxRowPos,maxColPos);
decodeLabels(birdKM3,maxRowPos,maxColPos);
decodeLabels(birdKM4,maxRowPos,maxColPos);
decodeLabels(birdKM5,maxRowPos,maxColPos);
decodeLabels(planeKM2,maxRowPos,maxColPos);
decodeLabels(planeKM3,maxRowPos,maxColPos);
decodeLabels(planeKM4,maxRowPos,maxColPos);
decodeLabels(planeKM5,maxRowPos,maxColPos);
%% GMM based clustering
birdGMM2 = hw4GMM(birdFeatures,2);
birdGMM3 = hw4GMM(birdFeatures,3);
birdGMM4 = hw4GMM(birdFeatures,4);
birdGMM5 = hw4GMM(birdFeatures,5);

planeGMM2 = hw4GMM(planeFeatures,2);
planeGMM3 = hw4GMM(planeFeatures,3);
planeGMM4 = hw4GMM(planeFeatures,4);
planeGMM5 = hw4GMM(planeFeatures,5);
%% Decode GMM to Images
decodeLabels(birdGMM2,maxRowPos,maxColPos)
decodeLabels(birdGMM3,maxRowPos,maxColPos)
decodeLabels(birdGMM4,maxRowPos,maxColPos)
decodeLabels(birdGMM5,maxRowPos,maxColPos)

decodeLabels(planeGMM2,maxRowPos,maxColPos)
decodeLabels(planeGMM3,maxRowPos,maxColPos)
decodeLabels(planeGMM4,maxRowPos,maxColPos)
decodeLabels(planeGMM5,maxRowPos,maxColPos)
