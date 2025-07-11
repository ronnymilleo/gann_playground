# Configuration Examples for Genetic Neural Network Game

## All Easy Modifications are now in globals.py! 

### 🚀 Quick Test Configuration (Fast)
```python
# Small population for quick testing
population = 20
generations = 25

# Smaller neural network (faster training)
h1_size = 6
h2_size = 4

# More exploration (higher mutation)
mutation_rate = 0.2  # 80% mutation chance
```

### 🧠 Smart Configuration (Balanced)
```python
# Moderate population
population = 50
generations = 100

# Standard network size
h1_size = 8
h2_size = 6

# Balanced mutation
mutation_rate = 0.4  # 60% mutation chance
```

### 🏆 High Performance Configuration (Slow but Thorough)
```python
# Large population for better diversity
population = 100
generations = 200

# Bigger neural network (more learning capacity)
h1_size = 16
h2_size = 12

# Fine-tuned mutation
mutation_rate = 0.3  # 70% mutation chance
mutation_strength_kernel = 2.0  # Stronger mutations
mutation_strength_bias = 0.8
```

### 🔬 Research Configuration (Experimental)
```python
# Very large population
population = 200
generations = 500

# Deep network
h1_size = 24
h2_size = 18

# Conservative mutation for stable evolution
mutation_rate = 0.6  # 40% mutation chance
mutation_strength_kernel = 1.0  # Gentler mutations
mutation_strength_bias = 0.3
```

## How to Apply:

1. **Edit `globals.py`** - Change any parameters you want
2. **Run `python main.py`** - Changes take effect immediately
3. **Experiment!** - Try different combinations

## Pro Tips:

- **Higher population** = Better diversity but slower evolution
- **Lower mutation_rate** = More mutations = More exploration
- **Bigger network (h1_size, h2_size)** = More learning capacity but slower training
- **Higher mutation_strength** = Bigger genetic changes per mutation

All parameters are now centralized in one file for easy experimentation!
