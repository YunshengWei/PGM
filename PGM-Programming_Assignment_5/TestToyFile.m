 % Copyright (C) Daphne Koller, Stanford University, 2012

 % modified
 % Based on scripts by Binesh Bannerjee (saving to file)
 % and Christian Tott (VisualizeConvergence).
 % Additionally plots a chart with (possibly inadequate) error score
 % relative to all exact marginals

 % These scripts depend on PA4 files for exact inference. Either
 % copy them to current dir, or add:
 % addpath '/path/to/your/PA4/Files'

 % If you notice that your MCMC wanders in circles, it may be
 % because rand function included with the assignment is still buggy
 % in your course run. In this case rename rand.m and randi.m
 % to some other names (like rand.bk and randi.bk). Don't forget to
 % rename them back if you are going to run test or submit scripts
 % that depend on them.


rand('seed', 1);

% Tunable parameters
num_chains_to_run = 3;
mix_time = 400;
collect = 6000;
on_diagonal = 1;
off_diagonal = 0.2;

% Directory to save the plots into, change to your output path
plotsdir = './plots_test';

start = time;

% Construct the toy network
[toy_network, toy_factors] = ConstructToyNetwork(on_diagonal, off_diagonal);
toy_evidence = zeros(1, length(toy_network.names));
%toy_clique_tree = CreateCliqueTree(toy_factors, []);
%toy_cluster_graph = CreateClusterGraph(toy_factors,[]);

% Exact Inference
ExactM = ComputeExactMarginalsBP(toy_factors, toy_evidence, 0);
graphics_toolkit('gnuplot');
figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, ExactM, 1, 'Exact');
print('-dpng', [plotsdir, '/EXACT.png']);

% Comment this in to run Approximate Inference on the toy network
% Approximate Inference
% % ApproxM = ApproxInference(toy_cluster_graph, toy_factors, toy_evidence);
% figure, VisualizeToyImageMarginals(toy_network, ApproxM);
% ^^ boobytrap, don't uncomment

% MCMC Inference
transition_names = {'Gibbs', 'MHUniform', 'MHGibbs', 'MHSwendsenWang1', 'MHSwendsenWang2'};
errors = {};

total_cycles = length(transition_names) * num_chains_to_run;
cycles_so_far = 0;
for j = 1:length(transition_names)
    samples_list = {};
    errors_list = [];

    for i = 1:num_chains_to_run
        % Random Initialization
        A0 = ceil(rand(1, length(toy_network.names)) .* toy_network.card);

        % Initialization to all ones
        % A0 = i * ones(1, length(toy_network.names));

        MCMCstart = time;
        [M, all_samples] = ...
            MCMCInference(toy_network, toy_factors, toy_evidence, transition_names{j}, mix_time, collect, 1, A0);
        samples_list{i} = all_samples;
        disp(['MCMCInference took ', num2str(time-MCMCstart), ' sec.']);
        fflush(stdout);
        errors_list(:, i) = CalculateErrors(toy_network, ExactM, all_samples, mix_time);
	err_start = time;
	disp(['Calculating errors took: ', num2str(time-err_start), ' sec.']);
	fflush(stdout);

        figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, M, i, transition_names{j}); 
        print('-dpng', [plotsdir, '/GREY_', transition_names{j}, '_sample', num2str(i), '.png']);

        cycles_so_far = cycles_so_far + 1;
        cycles_left =  (total_cycles - cycles_so_far);
	timeleft = ((time - start) / cycles_so_far) * cycles_left;
        disp(['Progress: ', num2str(cycles_so_far), '/', num2str(total_cycles), ...
              ', estimated time left to complete: ', num2str(timeleft), ' sec.']);
  
    end
    errors{j} = errors_list;

    vis_vars = [3];
    VisualizeMCMCMarginalsFile(plotsdir, samples_list, vis_vars, toy_network.card(vis_vars), toy_factors, ...
      500, ExactM(vis_vars),transition_names{j});
    VisualizeConvergence(plotsdir, samples_list, [3 10], ExactM([3 10]), transition_names{j});

    disp(['Saved results for MCMC with transition ', transition_names{j}]);
end

VisualizeErrors(plotsdir, errors, mix_time, transition_names);

elapsed = time - start;

fname = [plotsdir, "/report.txt"];
file = fopen(fname, 'a');
fdisp(file, ['On diag: ', num2str(on_diagonal),
             'Off diag: ', num2str(off_diagonal),
             'Mix time: ', num2str(mix_time),
             "Collect: ", num2str(collect),
             "Time consumed: ", num2str(elapsed), " sec."]);
fclose(file);

disp(["Done, time consumed: ", num2str(elapsed), " sec."]);
