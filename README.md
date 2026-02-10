# 🚗 Autonomous Driving with Reinforcement Learning

Projet étudiant d'apprentissage par renforcement appliqué à la conduite autonome dans le simulateur CARLA.

**Auteurs** : Jean Roelens, Axelle Refeyton, Yann Nguyen  
**Année** : 2025-2026

## 📋 Description

Ce projet explore différents algorithmes de RL pour entraîner un agent à conduire de manière autonome en environnement urbain. Trois approches ont été testées :

| Algorithme | Type | Résultat |
|------------|------|----------|
| DQN | Actions discrètes | Plafond de verre |
| PPO | Actions continues | Effondrement après ~200 épisodes |
| SAC | Actions continues | Meilleurs résultats, instable avec trafic |

Seul PPO est ici disponible.

## 🛠️ Prérequis

- Python 3.10+
- [CARLA Simulator 0.9.15](https://carla.org/)
- GPU compatible (attention aux problèmes AMD/Vulkan)

## 📦 Installation
```bash
git clone https://github.com/JeanRoelensYnov/Project_RL.git
cd Project_RL
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Lancer CARLA
```bash
# Windows
cd path/to/CARLA_0.9.15/WindowsNoEditor
CarlaUE4.exe
```

### 2. Tester la connexion
```bash
python test_connection_only.py
```

### 3. Entraîner l'agent PPO
```bash
python train.py
```

### 4. Tester un modèle entraîné
```bash
python test_PPO.py
```

## 📁 Structure du projet
```
Project_RL/
├── Documentation/      # Rapport et sujet du projet
├── logs/              # Logs d'entraînement (CSV)
├── models/            # Modèles sauvegardés (.pth)
├── carla_env.py       # Environnement Gym pour CARLA
├── config.py          # Configuration (carte, port, hyperparamètres)
├── PPO.py             # Implémentation PPO
├── train.py           # Script d'entraînement
├── debug_agent.py     # Test manuel de l'agent
└── test_*.py          # Scripts de test
```

## ⚙️ Configuration

Modifier `config.py` pour ajuster :
```python
CARLA_PATH = "D:\Code\Carla\CARLA_0.9.15\WindowsNoEditor"
CARLA_PORT = 2000
CARLA_MAP = "Town01"
TOTAL_TIMESTEPS = 100_000
```

## ⚠️ Problèmes connus

- Compatibilité limitée avec les GPU AMD (Vulkan)
- CARLA peut être instable selon la configuration matérielle
