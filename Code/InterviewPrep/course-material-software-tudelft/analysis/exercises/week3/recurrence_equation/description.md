Given below is a recurrence equation with its corresponding base case.

$$
T(0) = c_1 \\\\
T(g) = 3T(g - 1) + c_2
$$

You are asked to derive the closed form, prove that this is the correct closed form, and finally give the tightest runtime complexity in Big-Oh notation.

- To derive the closed form, you have to repeatedly unfold the recurrence equation.
  For this part, we suggest to at least show two steps of unfolding, e.g. at least for \\(T(g-1)\\) and \\(T(g-2)\\). 
- To prove that you indeed derived the correct closed form, you have to do a proof by induction.
  Make sure to be clear in your proofs.
- To give the tightest runtime complexity in Big-Oh notation, you should give a short informal argument on why your answer is correct.
  A full proof is not needed for this step.
