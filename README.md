# Zelda_RL

Zelda_RL is a reinforcement learning project that trains an AI agent to play **The Legend of Zelda: Link's Awakening DX** using **Proximal Policy Optimization (PPO)**. This project uses Python along with `stable-baselines3` to create, train, and evaluate the agent.

---

## 🚀 Features

- 🎮 Custom Gym environment for Zelda gameplay
- 🧠 PPO-based reinforcement learning agent
- 💾 Model saving and loading for continued training
- 📊 Training visualization and logging
- 🕹️ Emulator integration via `pyboy`
- 📹 Video recording of gameplay sessions
- 🎯 Reward system for exploration and combat
- 💪 Multi-process training support

---

## 📂 Installation

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/mtr26/Zelda_RL.git
cd Zelda_RL
```

### 2️⃣ Create a virtual environment (optional but recommended):
```bash
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3️⃣ Install dependencies:
```bash
pip install -r Rom/requirements.txt
```


## 🎮 Usage

### Training

```bash
python Train.py --timesteps 1000000 --num_cpu 10 --log_dir "tmp/"
```

### Testing 

```bash
python Test.py --timesteps 10000 --model_path "best_model"
```

