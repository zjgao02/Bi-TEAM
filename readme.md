# Bi-TEAM

Bi-TEAM is the first unified framework explores cross-scale biochemical space to predict the properties of NNAA-containing peptides 

![Bi-TEAM]{imgs/frame.png}
## Getting Started

### Prerequisites

- Python 3.10 or higher
- Required Python packages (see Installation section)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/zjgao02/Bi-TEAM.git
cd Bi-TEAM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training

To train a model using the default configuration:

```bash
./train.sh
```

You can customize training parameters in `config/config.py` or by modifying the training script.

#### Inference

To run inference with a trained model:

```bash
./inference.sh
```

## Supported Datasets

The framework comes with several pre-processed datasets:

- **hemo.csv**: Hemolysis data
- **human.csv**: Non-fouling data
- **ncaa.xlsx**: NNAA mapping dictionary
- **pampa.csv**: Parallel Artificial Membrane Permeability Assay data
- **Rezai.csv**: External dataset (details in paper)
- **solubility.csv**: Solubility data

## Customization

You can extend the framework by:

1. Adding new datasets to the `data` directory
2. Creating new model architectures in `model/models.py`
3. Modifying data preprocessing in `utils/data_utils.py`
4. Adjusting configuration parameters in `config/config.py`


## Citation
If you use Bi-TEAM in your research, please cite our paper:

```bibtex
@article{zhang2025sagephos,
  title={SAGEPhos: Sage Bio-Coupled and Augmented Fusion for Phosphorylation Site Detection},
  author={Zhang, Jingjie and Cao, Hanqun and Gao, Zijun and Wang, Xiaorui and Gu, Chunbin},
  journal={arXiv preprint arXiv:2502.07384},
  year={2025}
}
```

## Contact

If you have any questions, please feel free to contact the authors.

- Zijun Gao (zjgao24@cse.cuhk.edu.hk)