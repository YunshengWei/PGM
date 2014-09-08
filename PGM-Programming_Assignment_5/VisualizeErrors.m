
function VisualizeErrors(plotsdir, errors, mix_time, transition_names) 

% Draws a chart with cumulative error score of approx marginals relative to exact 
% marginals. 
% granularity - param to specify how often to extract approx marginals (1 = each iteration, smoother but slower)

  legendStrings = {};
  colors = {};
  cnt = 1;
  knowncolors = {"r", "g", "b", "m", "c", "r", "g", "b", "m", "c"};
  knownlinestyles = {"-", "--", ":", "-.", "-", "--", ":", "-." };
  er = errors{1};
  [s, c] = size(er);
  avg = zeros(s, 1);
  var = zeros(s, 1);
  for i = [1:length(errors)]
    er = errors{i}; 
    for j = [1:c]
      legendStrings{cnt} = [transition_names{i}, ' ', num2str(j)];
      cnt = cnt + 1;
    end
    av = sum(er') ./ c;
    v = sum(er' .^ 2) / c - (av .^2);
    avg(:,i) = av;
    var(:,i) = v;
    legendStrings{cnt} = [transition_names{i}, '  '];
    cnt = cnt + 1;
  end

% avg chart
  figure('visible', 'off');
  m = 0;
  set(gcf,'DefaultAxesColorOrder', rand(length(transition_names) * length(errors(1)), 3));
  for i = [1:length(errors)]
    transition = errors{i};
    [r c] = size(transition);
    for j = [1:c]
      options = [knowncolors{i}, ":"];
      plot(1:length(transition(:,j)), transition(:,j), options);
      if (max(transition(:,j) > m))
        m = max(transition(:,j));
      end;
      hold on;
    end
    plot(1:length(avg(:,i)), avg(:,i), [knowncolors{i}, "-"], 'LineWidth', 2);
    hold on;
  end
  % draw a vertical line at mix_time position
  x = linspace(mix_time, mix_time);
  y = linspace(0, m);
  plot(x, y, 'Linewidth',3)
  title(['Relative error. Average and chains ']);
  legend(legendStrings, "location", "southwest");
  grid on;
  print('-dpng', '-F:10', '-S1280,600', [plotsdir, '/ERRORS_AVG_MARGINAL.png']);

% var chart
  figure('visible', 'off');
  m = 0;
  mn = 1;
  set(gcf,'DefaultAxesColorOrder', rand(length(transition_names) * length(errors(1)), 3));
  for i = [1:length(errors)]
    transition = errors{i};
    if (max(var(:,i) > m))
      m = max(var(:,i));
    end;
    if (min(var(:,i) < m))
      mn = min(var(:,i));
    end;
    plot(1:length(var(:,i)), var(:,i), [knowncolors{i}, "-"], 'LineWidth', 2);
    hold on;
  end
  % draw a vertical line at mix_time position
  x = linspace(mix_time, mix_time);
  y = linspace(mn, m);
  plot(x, y, 'Linewidth',3)
  title(['Relative error variance']);
  legend(transition_names, "location", "southwest");
  grid on;
  print('-dpng', '-F:10', '-S1280,600', [plotsdir, '/ERRORS_VAR_MARGINAL.png']);

end

