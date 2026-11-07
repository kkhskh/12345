# Reviewer B — causal inference

Same-context pass, not an external referee.

The intervention is a hard replacement of one residual token followed by a full recomputation. That is a well-defined atomic intervention. The jointly reachable set is the image of that intervention, so the dependence is causal in the structural-equation sense: `b` is downstream of `a` and of `δ`.

The identification claim is narrower than it first sounds. The signs are properties of a weight class that was chosen so the line would be invariant. They are identifiable without the target response, and they are not identifiable for a generic initialization by the same short check. "Independently identified" here means "read off these weights," not "discovered in an unknown network without using the response."

Nothing in the gap is an effect on a decision. The readout is a fixed linear functional. Two steps on the same frozen weights change the sign of the raw response, from about `-0.233` at step `0.01` to about `0.819` at step `1`. The mathematics does not say which of those outcomes the model prefers, because there is no candidate set.

I accept the gap as a statement about responses. I do not accept a causal-discovery interpretation.
