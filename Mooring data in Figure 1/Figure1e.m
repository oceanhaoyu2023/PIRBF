clc;clear;
load data_1e.mat
time = 1:1:size(RBRdata_interp,2);
depth = 1:1:size(RBRdata_interp,1);
[depth,time] = meshgrid(depth,time);
figure
contourf(time,-depth,RBRdata_interp','linewidth',0.5,'showtext','off','linestyle','none')
xlim([min(time(:, 1))   max(time(:, 1))]);
ylim([min(-depth(1, :))  max(-depth(1, :))]);
colormap(jet)
colorbar
hold on
contour3(time,-depth,RBRdata_interp','LevelList',27,'color','k','linewidth',1)
xticks([1,721,1441,2161,2881,3601,4321,5040]);
xticklabels({'10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00'});
xlabel('Time')
ylabel('Depth')

