function F2 = ReorderFactorVars(F, newVarOrder)
  % input:
  % F is a factor with fields: var, card, and val
  % newVarOrder is a reordering off the F.card array
  %   so if F.card = [2,5,7], then [2,5,7] or [7,2,5] etc are legal
  %   values for newVarOrder, while [2,5] or [2,5,1] etc are not legal
  %
  % output F2 is basically the same as F execept its var and card
  % now match the reordered variables, and the .val array has been
  % reordered to fit with the changed variables
  %
  % Thanks and credit to Alex Gilman for the [~,permute] in line 18 part
  % Use this at your own risk. I'd suggest trying it out on
  % some test factors (like FullI.RandomFactors(1) ) to see what it does

  assert(length(F.var) == length(newVarOrder));
  [test, permute] = ismember(F.var, newVarOrder);
  assert(prod(test) ~= 0); % if passed, newVarOrder is valid perm. of F.var,
  [~, newP] = ismember(newVarOrder,F.var); % newP is the positional permutation of F with inverse permute
  F2.var = newVarOrder;
  F2.card = F.card(newP);
  % to match, we take F2's assignment table and reorder columns by permute
  newAssign = (IndexToAssignment([1:prod(F2.card)],F2.card));
  newAssign = newAssign(:,permute);
  % now we use F's card to evaluate each row of newAssign as an index into F.val
  idx = AssignmentToIndex(newAssign,F.card);
  F2.val = F.val(idx);
end