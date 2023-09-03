# torch2circom

![torch2circom_banner](https://github.com/torch2circom/torch2circom/assets/87213416/bbd40141-7b85-4ca9-8a60-6b045644201f)

torch2circom is a python tool that transpiles a PyTorch model into a Circom circuit.

## Installation

First, clone the repository:

```bash
git clone https://github.com/torch2circom/torch2circom.git
```

Then, install the dependencies. You can use pip:

```bash
pip install -r requirements.txt
```

If you use conda, you can also create a new environment with the following command:

```bash
conda env create -f environment.yml
```

You will also need to install circom and snarkjs. You can run the following commands to install them:

```bash
bash setup-circom.sh
```

Last but not least, run

```bash
npm install
```

## Testing

To test the package, you can run the following command:


```bash
npm test
```

## Usage

To use the package, you can run the following command:

```bash
python main.py <model_path> [-o <output_dir>] [--raw]
```

For example, to transpile the model in `models/best_practice.h5` into a circom circuit, you can run:

```bash
python main.py models/best_practice.h5
```

The output will be in the `output` directory.

If you want to transpile the model into a circom circuit with "raw" output, i.e. no ArgMax at the end, you can run:

```bash
python main.py models/best_practice.h5 --raw
```

## Acknowledgements
This project is based on [keras2circom](https://github.com/socathie/keras2circom) by [@socathie](https://github.com/socathie).