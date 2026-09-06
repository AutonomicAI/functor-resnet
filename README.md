# Functor ResNet

A research/demo implementation of a Functor-style ResNet model with benchmark support.

This repository explores a **Functor Model** approach to neural-network updates: instead of treating a model as a monolithic mutable object, the model is represented as an initial executable function plus a sequence of auditable functional deltas.

The goal is to support:

- Streaming model updates
- Versioned model reconstruction
- Auditability of model changes
- Rollback to prior versions
- Benchmarking of functional-update behavior
- Separation of model semantics from serving/demo infrastructure

## Concept

In a production deployment, model update events could be written to an append-only event stream such as Kafka.

A complete model version can then be reconstructed from:

1. The initial model function
2. The ordered stream of delta changes
3. Any associated weight or parameter changes
4. The validation/audit metadata for each committed update

This makes the model lifecycle closer to event-sourced software architecture than traditional opaque checkpoint replacement.

## Repository Structure

```text
slm_template/
  components/
    f.py        # Core model function
    ...         # Supporting extracted components

README.md

## Benchmarking 
See [Claude's report](https://www.functormodel.ai/benchmark)
