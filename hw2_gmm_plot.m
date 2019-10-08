function hw2_gmm_plot(sample1,sample2,t)
    classes = ['Class 1';'Class 2'];    
    hold on    
    scatter(sample1(:,1),sample1(:,2),'kx')
    scatter(sample2(:,1),sample2(:,2),'r+')
    limits = [-4,8];
    ylim(limits)
    xlim(limits)
    title(t)
    xlabel('x1')
    ylabel('x2')
    legend(classes)
end