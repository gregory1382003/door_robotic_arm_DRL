# _robotic_arm_DRL -
Robotic arm manipulation using deep reinforcement learning

## Setup
Create and activate a conda environment:
```bash
conda create -n arm-rl python=3.9
conda activate arm-rl
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training
Navigate to your project directory and Run training:
```bash
python main.py
```
View training metrics in TensorBoard (logs are written to `./logs/`):
```bash
tensorboard --logdir logs --port 6006
```
Then open `http://localhost:6006` in your browser.

## Demo
Run the demo (opens an on-screen render window):
```bash
python test.py
```
