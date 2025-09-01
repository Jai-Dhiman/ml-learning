# My Piano Performance Analysis

**Personal recreation of PercePiano dataset analysis for deep learning fundamentals**

## Project Goal

Build piano performance analysis system from scratch to predict 19 perceptual dimensions:

- Timing, Articulation, Pedal, Timbre, Dynamics, Musical Expression, Emotion, Interpretation

## Approach

**Phase 1**: Dataset Recreation & ML Fundamentals (Current)

- ✅ Dataset exploration and correlation analysis
- ✅ Audio preprocessing pipeline
- 🚧 Feature extraction matching research approaches
- 🎯 Single-dimension prediction model (next)

**Future Phases**: CNN architecture → Multi-task learning → Chopin/Liszt extension

## Repository Structure

```
my_implementation/
├── src/           # Clean implementation code
├── data/          # PercePiano labels and sample audio
├── notebooks/     # Jupyter experiments 
├── models/        # Trained model files
├── experiments/   # Training configurations
└── results/       # Analysis outputs
```

## Quick Start

```bash
cd src/
python dataset_analysis.py      # Analyze PercePiano dataset
python audio_preprocessing.py   # Test audio feature extraction
```

## Key Insights from Dataset Analysis

- **1202 performances** across 19 perceptual dimensions
- **22 performers**, multi-composer dataset (Schubert, Beethoven)  
- **Rating range**: [0.143, 0.976], mean=0.553
- **Strong correlations**: Related dimensions (pedal types, musical expression)
- **Audio features**: Tempo→timing, spectral centroid→brightness, RMS→dynamics

## Next Steps

1. Build single-dimension prediction model (starting with "Timing_Stable_Unstable")
2. Implement multi-task learning for all 19 dimensions
3. Compare CNN vs traditional ML approaches
4. Extend to new repertoire (Chopin/Liszt recordings)

---
*Learning-focused implementation - building everything from scratch for deep understanding*
