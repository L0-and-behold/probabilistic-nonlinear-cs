# Probabilistic and Nonlinear Compressive Sensing

This repository contains experimental implementations and analysis code for research on probabilistic and nonlinear compressive sensing methods in neural network training.

## Repository Structure

The repository is organized into two main experimental directories:

### EGP/

Contains an implementation of Exact Gradient Pruning (EGP) with benchmarks against other compressive sensing techniques. 

It employs a probabilistic reformulation that is drastically faster than Monte Carlo based methods, while exhibiting lower test loss than conventional compressive sensing algorithms.

See [`EGP/README.md`](EGP/README.md) for detailed setup instructions and experiment configuration.

### l2-experiments/

Contains code for investigating the _ℓ₂-distance rebound phenomenon_, where student networks initially approach but then progressively diverge from teacher network parameters during training, even while improving functionally.

See [`l2-experiments/README.md`](l2-experiments/README.md) for detailed setup instructions and experiment configuration.


## Getting Started

Each experimental directory is designed to be used as an independent root for running experiments:

1. __For EGP experiments__: Navigate to `EGP/` and follow the setup instructions in its README
2. __For ℓ₂-experiments__: Navigate to `l2-experiments/` and follow the setup instructions in its README

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{barth2025probabilistic_and_nonlinear_compressive_sensing,
      title={Probabilistic and nonlinear compressive sensing}, 
      author={Lukas Silvester Barth and Paulo {von Petersenn}},
      year={2025},
      eprint={2509.15060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.15060}, 
}
```

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).

