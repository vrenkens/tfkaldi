load('alpha.mat')


figure
surf(alpha)
%shading('flat')

out = {'<sos>', 'sil','hh','ih','f','sil','k','er','r','ow','ow','sil','sil','t','ah','m','aa','hh','hh','ae','v','er','r','r','n','n','sil','f','er','er','m','iy','iy','iy','iy','iy','iy','iy','iy','sil','sil','t','uw','sil','<eos>'};
set(gca, 'XTick',1:45, 'XTickLabel', out(1:end))
view(90, 90)
%view(395, 45)
colormap('jet')
%shading('flat')