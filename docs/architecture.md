# HEIDI architecture boundary

The public implementation describes the following computational path:

1. Current node values and observation masks are embedded together with global
   context and optional time information.
2. Edge-conditioned directed message passing propagates information through
   the production graph.
3. A forward-only temporal recurrence produces causal graph-temporal states.
4. Bayesian heads infer a latent posterior and emit reconstruction and forecast
   distributions.
5. Freshly observed values are preserved; only unavailable values are replaced
   by the inferred reconstruction.
6. Reconstruction and one-step forecast terms can be combined with latent and
   Bayesian KL terms through the public loss interface.

The release intentionally leaves the following experiment-specific choices to
the maintainer: feature schema, dimensions, layer count, optimizer and
schedule, loss weights, split construction, missingness protocol, seeds,
normalization statistics, checkpoint selection, and evaluation sampling count.
Those values are not encoded as defaults in the public model constructor.

See [`HEIDI_architecture.pdf`](HEIDI_architecture.pdf) for the conceptual
figure.
