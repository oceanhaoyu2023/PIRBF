clear;
clc;

load('data_1d.mat')

figure
contourf(times,depth,Temp,'linewidth',1,'showtext','off','linestyle','none')
shading interp
hold on
contour(times,depth,Temp,[25,25],'color','black','linewidth',1);
datetick('x',15);axis tight;
ylim([-140,-40])
colormap(jet); colorbar;
set(gca,'fontsize',14)
xlabel('Time')
ylabel('Depth(m)')
set(gca,'xtick', times(1):6*0.0069*3:times(end) )
set(gca,'xticklabels', {'00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','24:00',} )

