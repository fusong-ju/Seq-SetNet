# Seq-SetNet
## About The Project
The implementation of the paper "Seq-SetNet: directly exploiting multiple sequence alignment for
protein structure prediction".

## Getting Started
### Prerequisites
Install [PyTorch 1.4+](https://pytorch.org/)

### Installation

1. Clone the repo
```sh
git clone https://github.com/fusong-ju/Seq-SetNet.git
```

## Usage
1. Generate `a3m` format MSA for a given target sequence using hhblits
2. Run Seq-SetNet
```sh
run_inference.sh <MSA> <out_path>
```

## Example
```sh
cd example
./run_example.sh
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
