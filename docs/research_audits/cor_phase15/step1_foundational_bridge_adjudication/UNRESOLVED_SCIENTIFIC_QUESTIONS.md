# Unresolved scientific questions

1. **Agreement computed nonlinearly.** Does any trained Transformer contain a component (an MLP neuron, or a QK composition) whose output is nonlinear *in an agreement variable* between two evidence sources, with salience held fixed? An example would be a neuron that responds to the product of two evidence features. This step did not examine this class. It is the only class left in which a consistency-specific mechanism could exist.
2. **QK composition.** One source can change another source's key or query salience. That is the reallocation term of CI.3* produced by computation rather than by token design. Its sign structure was not analyzed.
3. **Deep circuits.** How often do the σ channel and downstream re-readers (backup and negative heads, erasure neurons) reverse a head's direct term in trained models? This decides whether any head-level result transfers to decisions. Rushing and Nanda (2024) report that self-repair is noisy and sometimes overcorrects.
4. **Effective null space.** NC was proved for the exact null space. The effective, low-rank null space of real `W_U` (Stolfo et al. 2024) needs perturbation bounds on single crossing and on the `κ` limit.
5. **RB for families.** A metric-free combination rule for per-context RB tests across prompts. Order-restricted inference is the relevant literature.
6. **RMSNorm models.** Norm-only edits cannot change any ranking. Whether any other relation-sensitive channel exists there beyond softmax is open.
