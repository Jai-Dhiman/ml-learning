# [piano music] - Deep Learning Audio Analysis Tasks

*Last Updated: 2025-08-22*

## Current Sprint: Phase 1 - PercePiano Dataset Recreation & ML Fundamentals (Months 1-3)

### In Progress

- [ ] CNN architecture for piano analysis
  - Study CNN fundamentals: convolution, pooling, feature maps
  - Implement CNN on mel-spectrograms (2D audio representations)
  - Compare CNN vs feedforward network performance
  - **Acceptance**: CNN model that processes audio spectrograms effectively

### Todo

#### Week 1-2: PercePiano Dataset Exploration

- [ ] Set up development environment for PercePiano analysis
  - Install PyTorch, librosa, matplotlib, numpy, pandas
  - Create Jupyter notebook environment for dataset exploration
  - Load sample PercePiano audio files and labels successfully
  - **Acceptance**: Can load and visualize PercePiano audio files with their ratings

- [ ] Deep dive into PercePiano annotation system
  - Study all 19 perceptual dimensions from the paper
  - Load and explore the label files (JSON format with ratings)
  - Visualize distribution of ratings across different dimensions
  - Identify which dimensions have the most/least variation
  - **Acceptance**: Clear understanding of PercePiano's perceptual evaluation framework

- [x] Audio-visual correlation analysis (2025-08-25)
  - ✓ Created audio loading pipeline with librosa
  - ✓ Built feature extraction for tempo, spectral features, dynamics  
  - ✓ Analyzed correlation between audio features and perceptual ratings
  - ✓ Identified extreme performance examples (lowest: 0.350, highest: 0.726)
  - ✓ Found key correlations: timing→tempo stability, brightness→spectral centroid, dynamics→RMS range
  - ✓ **Acceptance**: Successfully identified audio patterns that correlate with perceptual ratings

#### Week 3-4: Recreation of Basic Feature Extraction

- [x] Implement PercePiano audio preprocessing pipeline (2025-08-25)
  - ✓ Built standardized audio loading with consistent 22.05kHz sampling rate
  - ✓ Implemented spectrograms, mel-spectrograms (128 bands), MFCCs (13 coefficients)  
  - ✓ Added chromagram, tempo, onset detection, spectral features (18 total)
  - ✓ Created robust preprocessing class for batch processing
  - ✓ **Acceptance**: Successfully processes PercePiano audio files consistently

- [x] Build feature extraction matching original approach (2025-08-25)
  - ✓ Studied VirtuosoNet (CNN on note embeddings) and MidiBERT (transformer on MIDI tokens)
  - ✓ Created clean workspace in my_implementation/ separate from original PercePiano repo
  - ✓ Implemented audio-based feature extraction: mel-spectrograms, MFCCs, chromagrams
  - ✓ Built 10 scalar features for correlation analysis: tempo, spectral, dynamic features
  - ✓ **Acceptance**: Clean feature extraction pipeline optimized for piano audio analysis

- [x] Neural network for single perceptual dimension prediction (2025-08-25)
  - ✓ Built simple feedforward network: 10 features → 32 → 16 → 1 output
  - ✓ Implemented complete training loop with Adam optimizer, MSE loss, early stopping
  - ✓ Achieved 0.357 correlation on timing prediction with validation/test splits
  - ✓ Generated learning curves and model performance analysis
  - ✓ **Acceptance**: Working model predicts timing dimension with good correlation

- [x] Build multi-task neural network for all 19 dimensions (2025-08-25)
  - ✓ Implemented shared feature extraction with task-specific heads (13,331 parameters)
  - ✓ Trained on all perceptual dimensions simultaneously with multi-output architecture
  - ✓ Analyzed task relationships and correlation patterns across dimensions
  - ✓ Best performance: Dynamic Range (0.312), Energy (0.215), Timbre (0.182)
  - ✓ **Acceptance**: Multi-task model successfully predicts all 19 dimensions with average 0.086 correlation

#### Month 2: Multi-Task PercePiano Modeling

- [x] Build multi-task neural network for all 19 dimensions (2025-08-25)
  - ✓ Implemented shared feature extraction with 19 task-specific prediction heads
  - ✓ Handled multi-dimensional outputs with proper loss functions and evaluation
  - ✓ Compared single-task (0.357) vs multi-task (0.086 avg) learning performance
  - ✓ **Acceptance**: Multi-task model successfully predicts all PercePiano dimensions simultaneously

- [ ] Advanced feature engineering for piano performance
  - Extract tempo, rhythm, and timing-related features
  - Implement harmonic analysis (chord recognition, key detection)
  - Add performance-specific features (dynamics curves, articulation patterns)
  - **Acceptance**: Rich feature set that captures nuances of piano performance evaluation

- [ ] Model performance analysis and debugging
  - Analyze which dimensions are easiest/hardest to predict
  - Study correlation patterns between predicted and actual ratings
  - Identify failure cases and understand why they fail
  - **Acceptance**: Deep understanding of model strengths and limitations

#### Month 3: Recreation Validation and CNN Preparation

- [ ] Benchmark against original PercePiano results
  - Replicate the performance metrics from the original paper
  - Compare your feature extraction and modeling approach
  - Achieve comparable or better performance on key metrics
  - **Acceptance**: Results that validate your understanding of the dataset

- [ ] Comprehensive audio representation comparison
  - Test spectrograms, mel-spectrograms, CQT, chromagrams for different dimensions
  - Understand which audio representations work best for which perceptual aspects
  - Document trade-offs between different preprocessing approaches
  - **Acceptance**: Evidence-based recommendations for audio preprocessing

- [ ] Foundation for CNN implementation in Phase 2
  - Prepare spectrograms in CNN-ready format
  - Study successful CNN architectures for audio analysis
  - Plan how to incorporate CNNs into your multi-task framework
  - **Acceptance**: Clear technical roadmap for CNN implementation with PercePiano data

### Completed

- [x] Removed previous Random Forest approach and baseline implementation (2025-08-22)
- [x] Updated roadmap to focus on deep learning from first principles (2025-08-22)
- [x] Restructured tasks to emphasize understanding over quick results (2025-08-22)
- [x] Updated roadmap and tasks to focus on PercePiano dataset recreation approach (2025-08-25)
- [x] Explore and understand PercePiano dataset structure and annotations (2025-08-25)
  - ✓ Dataset contains 1202 performances across 19 perceptual dimensions
  - ✓ Multi-composer dataset: Schubert (964), Beethoven (238) performances  
  - ✓ 22 unique performers with varying numbers of recordings
  - ✓ Ratings normalized to [0-1] range with overall mean of 0.553
  - ✓ Strong correlations found between related dimensions (e.g., pedal types, musical balance)
  - ✓ Created comprehensive dataset analysis tools and insights
- [x] Set up development environment for PercePiano analysis (2025-08-25)
  - ✓ Python virtual environment with PyTorch, librosa, pandas, numpy
  - ✓ Jupyter notebook ready for interactive exploration
  - ✓ Dataset analysis scripts created and functional
- [x] Deep dive into PercePiano annotation system (2025-08-25)
  - ✓ 19 dimensions grouped into 7 categories: Timing, Articulation, Pedal, Timbre, Dynamic, Music Making, Emotion, Interpretation
  - ✓ Rating distributions analyzed - all dimensions show reasonable variation
  - ✓ Player analysis complete - balanced dataset across multiple performers
  - ✓ Musical repertoire analysis: primarily 8-bar segments from classical piano works

---

## Phase 2 Preview: CNN Architecture for Piano Analysis (Months 4-6)

### Upcoming Focus Areas

- **CNN Fundamentals**: Convolution, pooling, feature maps, translation invariance
- **Spectrogram CNNs**: Why CNNs work well on spectrograms, kernel design
- **Multi-task Learning**: Single model predicting timing, dynamics, articulation
- **Piano-Specific Architecture**: Design choices optimized for piano analysis

---

## Notes & Context

### Current Philosophy - Dataset-Driven Deep Learning

Working through fundamental ML concepts via PercePiano recreation with emphasis on:

1. **Learning by Recreation**: Understand concepts by reproducing research results
2. **Perceptual Grounding**: Every technical decision validated against human ratings
3. **Progressive Complexity**: Single dimension → multi-task → advanced architectures
4. **Musical Relevance**: Connect audio features to meaningful musical judgments

### Phase 1 Learning Goals (PercePiano Focus)

- **Dataset Mastery**: Deep understanding of PercePiano's 19 perceptual dimensions
- **Audio-to-Perception Pipeline**: Can extract features that predict human ratings
- **Multi-Task Neural Networks**: Handle correlated perceptual dimensions effectively
- **Benchmarking Skills**: Validate results against published research

### Key Resources for Phase 1

- **PercePiano Dataset**: 19-dimensional perceptual ratings with piano audio
- **Original Research Papers**: VirtuosoNet and MidiBERT-Piano implementations
- **Librosa + PyTorch**: Audio processing and neural network implementation
- **Human Perceptual Studies**: Understanding what makes piano performance evaluation difficult

### Success Metrics for Phase 1

- Recreated PercePiano preprocessing and achieved comparable benchmark results
- Built multi-task model that predicts all 19 perceptual dimensions
- Can identify which audio features correlate with which perceptual aspects
- Ready to improve performance with CNN architectures in Phase 2

---

## Task Management Reminders

1. **Focus on Understanding**: Prefer implementing from scratch over using pre-built solutions
2. **Document Learning**: Keep notes on insights and "aha moments"
3. **Experiment Actively**: Try variations and see how they affect results
4. **Connect to Music**: Always relate technical concepts to musical understanding
5. **Build Reusable Components**: Code written in Phase 1 will be used in later phases
6. **Time-Box Learning**: Set specific deadlines to avoid endless perfectionism
