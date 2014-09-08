
function errors = CalculateErrors(G, TrueMargs, all_samples, mix_time)
% Calculates error distance between all true marginals and samples marginals at each step
% Sampled marginals after mix_time will be calculated without the data before mix_time

  GetError = @GetError3;  % select which error metric should be used
  [nsamples, nvars] = size(all_samples);

%  all_m = {};
  
  EmptyM = repmat(struct('var', 0, 'card', 0, 'val', []), length(G.names), 1);
  for i = 1:length(G.names)
    EmptyM(i).var = i;
    EmptyM(i).card = G.card(i);
    EmptyM(i).val = zeros(1, G.card(i));
  end
  M = EmptyM;
  PrevM = EmptyM;

  for s = [1:nsamples]
    sample = all_samples(s,:);
    if (mix_time == s)      % reset the counters
      PrevM = EmptyM;
    end
    NormM = EmptyM;
    for j=1:length(sample)
      PrevM(j).val(sample(j)) = PrevM(j).val(sample(j)) + 1;
    end
    NormM = PrevM;
    for j= [1:length(sample)]
      NormM(j).val = NormM(j).val ./ s;
    end
%    all_m{s} = M;    % no need to store marginals on each step at this moment
    errors(s) = GetError(TrueMargs, NormM);
  end
end

% Error metric functions, possibly not quite adequate? L.F.
% change to whatever score measurement technique you like.


function er = GetError1(ExactM, M)
  er = norm([ExactM(:).val] - [M(:).val]) / norm([ExactM(:).val] + [M(:).val]);
end

function er = GetError2(ExactM, M)
  % play with setting the input mean (0.5), truncate all results above 1
  exact = [ExactM(:).val] .- 0.5;
  m = [M(:).val] .- 0.5;
  er = min([1 abs(norm(exact - m) / norm(exact + m))]);
end

function er = GetError3(ExactM, M)
  % linear absolute error
  m = [M(:).val];
  er = sum(abs([ExactM(:).val] - m)) / length(m);
end

