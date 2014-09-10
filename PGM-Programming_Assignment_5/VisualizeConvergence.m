function VisualizeConvergence(plotsdir, samples_list, V, ExactMarginals, tname) 
% Ignoring the cardinality (assumes card == 2)  -- only looking at P(v(i) = 2)
% The word "convergence" here is used imprecisely and is only meant to motivate the basic idea

legendStrings = {};
cnt = 1;
for i = 1:length(V),
  for j = 1:length(samples_list),
    legendStrings{cnt} = ['Chain ', num2str(j), ', Variable X', num2str(V(i))];
    cnt = cnt + 1;
  end
end

% This color scheme works in octave. For Matlab replace with
% colors = ['r', 'g', 'b', 'k', 'm', '--b']; 
colors = ['1', '2', '3', '4', '5', '6'];

figure('visible', 'off')

cnt = 0;
set(gcf,'DefaultAxesColorOrder', rand(length(V) * length(samples_list), 3));
for i = 1:length(V)
    v = V(i);
    for j = 1:length(samples_list)
        samples_v = samples_list{j}(:, v);
        M = ExactMarginals(i);
        dat = (cumsum(samples_v - 1)' ./ (1:length(samples_v))) - M.val(2);

        % if the following line fails in Matlab, replace it with
        % plot(1:length(dat), dat, colors(mod(cnt,6)+1),'LineWidth', 2);
        plot(1:length(dat), dat, 'LineWidth', 2, colors(mod(cnt,6)+1));
        cnt = cnt + 1;
        hold on;
    end
end

title(['Convergence plot for ' tname]);
legend(legendStrings);
grid on;
print('-dpng', [plotsdir, '/CONVERGENCE_', tname, '.png']);
