% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  D = I.DecisionFactors;
  euf = CalculateExpectedUtilityFactor(I);
  euf = ReorderFactorVars(euf, D.var);
  MEU = 0;
  OptimalDecisionRule = D;
  OptimalDecisionRule.val = zeros(1, prod(OptimalDecisionRule.card));
  % potential bug here!
  for i = 1:prod(D.card(2:end))
      head = (i - 1) * D.card(1) + 1;
      tail = i * D.card(1);
      [Y, I] = max(euf.val(head:tail));
      OptimalDecisionRule.val(I + head - 1) = 1;
      MEU = MEU + Y;
  end

end
