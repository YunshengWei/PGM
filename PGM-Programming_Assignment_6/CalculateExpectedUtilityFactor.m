% Copyright (C) Daphne Koller, Stanford University, 2012

function EUF = CalculateExpectedUtilityFactor( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: A factor over the scope of the decision rule D from I that
  % gives the conditional utility given each assignment for D.var
  %
  % Note - We assume I has a single decision node and utility node.
  EUF = repmat(struct('var', [], 'card', [], 'val', []), length(I.UtilityFactors), 1);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  %
  D = I.DecisionFactors;
  for k = 1:length(I.UtilityFactors)
      F = [I.RandomFactors, I.UtilityFactors(k)];
      mu = VariableElimination(F, setdiff([F.var], D.var));
      EUF(k) = struct('var', [], 'card', [], 'val', []);
      for i = 1:length(mu)
        EUF(k) = FactorProduct(EUF(k), mu(i));
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  


  
end  
