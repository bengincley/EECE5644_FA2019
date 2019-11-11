function decodeLabels(kMeanVector,maxRowPos,maxColPos)
    imgKM = reshape(kMeanVector,maxRowPos,maxColPos);
    imgKMnorm = (imgKM/max(imgKM,[],'all'));
    figure(); imshow(imgKMnorm)