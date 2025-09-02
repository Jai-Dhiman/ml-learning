# 🎹 Chopin/Liszt Data Collection Guide

## Audio Requirements (Based on CNN Architecture)

### Technical Specifications

- **Sample Rate**: 22.05 kHz (consistent with preprocessing pipeline)
- **Format**: WAV preferred, MP3 acceptable
- **Segment Length**: 3-4 seconds optimal (128 time frames @ 512 hop length)
- **Quality**: High bitrate recordings (>192kbps for MP3, >44.1kHz source for WAV)

### Target Dataset Size

- **Minimum**: 100 unique performances (viable dataset)
- **Target**: 300-500 performances (robust training)
- **Goal**: 10-15 different interpreters per piece

## Recording Sources & Quality Guidelines

### 1. YouTube Concert Recordings

**Good Sources:**

- Competition finals (Cliburn, Chopin Competition)
- Master class recordings (clear audio, minimal audience noise)
- Studio recordings posted by conservatories

**Quality Markers:**

- ✅ Clear piano sound, minimal reverb
- ✅ Consistent recording level
- ✅ No audience noise during performance
- ✅ Complete performances (avoid edited compilations)

**Avoid:**

- ❌ Phone recordings from audience
- ❌ Heavy compression/audio artifacts
- ❌ Recordings with talking over music
- ❌ Auto-generated/MIDI performances

### 2. IMSLP Public Domain

**Advantages:**

- Legal/copyright free
- Often high-quality historical recordings
- Diverse performance styles

**Limitations:**

- Older recording quality
- Limited repertoire overlap

### 3. Personal Recordings

**Include Your Own:**

- Different skill levels provide valuable data
- You can control recording quality
- Can target specific repertoire gaps

## Data Collection Protocol

### 1. Audio Extraction

```bash
# YouTube download (using yt-dlp)
yt-dlp -f "bestaudio[ext=m4a]" --extract-audio --audio-format wav [URL]

# Convert to target format
ffmpeg -i input.wav -ar 22050 -ac 1 output.wav
```

### 2. Segmentation Strategy

Based on PercePiano research findings:

```python
# Level 1: Short segments for technical features
short_segment_length = 3.0   # seconds - for timing, articulation
short_overlap = 0.5          # 50% overlap

# Level 2: Longer segments for musical interpretation  
long_segment_length = 12.0   # seconds - for phrasing, expression
long_overlap = 0.25          # 25% overlap

# Musical phrase boundaries (preferred when detectable):
# - 4 bars ≈ 8-12 seconds
# - 8 bars ≈ 15-25 seconds  
# - Use beat tracking to align with musical structure when possible
```

**Segmentation Priority:**

1. **Musical phrases** (when identifiable) - most meaningful units
2. **Fixed-length segments** - for consistent processing
3. **Overlap strategy** - ensures no musical content is lost

### 3. Quality Control Checklist

- [ ] Audio plays without artifacts
- [ ] Consistent volume level across segments
- [ ] No clipping or distortion
- [ ] Clear piano sound (not muffled/distant)
- [ ] Minimal background noise

## Labeling Strategy

### 1. Perceptual Dimensions (19 total)

**Hierarchical Organization (Based on PercePiano Research):**

**Low-Level Features (3-5 second segments):**

```python
Timing:
- Stable ←→ Unstable

Articulation:
- Short ←→ Long
- Soft-cushioned ←→ Hard-solid
```

**Mid-Level Features (8-12 second segments):**

```python
Pedal:
- Sparse-dry ←→ Saturated-wet
- Clean ←→ Blurred

Timbre:
- Even ←→ Colorful
- Shallow ←→ Rich
- Bright ←→ Dark
- Soft ←→ Loud

Dynamics:
- Sophisticated-mellow ←→ Raw-crude
- Little dynamic range ←→ Large dynamic range
```

**High-Level Features (12+ second segments):**

```python
Music Making:
- Fast-paced ←→ Slow-paced
- Flat ←→ Spacious
- Disproportioned ←→ Balanced
- Pure ←→ Dramatic-expressive

Emotion & Mood:
- Optimistic-pleasant ←→ Dark
- Low Energy ←→ High Energy
- Honest ←→ Imaginative

Interpretation:
- Unsatisfactory-doubtful ←→ Convincing
```

**Key Insights from PercePiano:**

- Use **bipolar scales** (7-point: 1=strongly Option A, 7=strongly Option B)
- Features organized by **analysis window length** needed for judgment
- Include "uncertain/don't know" option for difficult cases

### 2. Rating Interface Setup

```python
# Create simple web interface or Jupyter widget
import ipywidgets as widgets
from IPython.display import Audio, display

def create_rating_interface(audio_segment, segment_id):
    # Audio playback
    audio_player = Audio(audio_segment, rate=22050)
    
    # 19 sliders for dimensions
    sliders = {}
    for dim in dimension_names:
        slider = widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.1,
            description=dim[:20], layout=widgets.Layout(width='400px')
        )
        sliders[dim] = slider
    
    return audio_player, sliders
```

### 3. Validation Strategy (Based on PercePiano Findings)

**Primary Approach: Self-Evaluation with Validation**

PercePiano found individual musical perception is subjective, but consensus emerges through averaging multiple experts. For your project:

**Phase 1: Consistent Self-Annotation**

- Rate all segments yourself first for consistency
- Include 2-3 "anchor" segments per rating session for calibration
- Rate same segment twice (separated by sessions) - target >0.8 correlation

**Phase 2: External Validation**

- Recruit 2-3 other pianists to rate subset (20-30 segments)
- Compare their ratings with yours to identify systematic biases
- Focus validation on segments where you felt uncertain

**Quality Controls:**

- Include "uncertain/don't know" option (PercePiano: 7.3% of responses)
- Document detailed rating guidelines like PercePiano
- Take breaks between rating sessions to maintain consistency

## Implementation Timeline

### Week 1: Audio Collection

- [ ] Set up download tools (yt-dlp, ffmpeg)
- [ ] Collect 50 diverse Chopin/Liszt recordings
- [ ] Test preprocessing pipeline on sample recordings
- [ ] Document performer/piece metadata

### Week 2: Labeling Interface & Pilot

- [ ] Build rating interface (Jupyter widgets or web app)
- [ ] Rate 20-30 segments for interface testing
- [ ] Check consistency with repeat ratings
- [ ] Refine rating process based on pilot

### Week 3: Full Dataset Creation

- [ ] Collect remaining recordings (target 100+ performances)
- [ ] Complete perceptual labeling for all segments
- [ ] Quality control and consistency checking
- [ ] Export dataset in training format

### Week 4: Model Training

- [ ] Train CNN models on real Chopin/Liszt data
- [ ] Compare with synthetic data validation results
- [ ] Evaluate against PercePiano baseline (if available)
- [ ] Document performance improvements

## Expected Challenges & Solutions

### Challenge: Labeling Fatigue

**Solution**: Rate in 30-minute sessions, max 15 segments per session (PercePiano standard)

### Challenge: Consistency Across Sessions  

**Solution**: Include 2-3 "anchor" segments per session + uncertainty tracking

### Challenge: Subjectivity in Musical Perception

**PercePiano Finding**: Individual ratings have "poor" reliability, but averaged ratings show "excellent" reliability
**Solution**: Start with self-evaluation, then validate subset with other musicians

### Challenge: Segment Length vs. Feature Type

**New Challenge**: Different features need different analysis windows
**Solution**: Use hierarchical segmentation (3-5s for technical, 12s+ for interpretive)

### Challenge: Copyright Issues

**Solution**: Focus on competition recordings, IMSLP, and personal recordings

### Challenge: Performer Bias

**Solution**: Balance skill levels, include student + professional recordings

## Success Metrics

### Dataset Quality

- **Size**: 300+ labeled segments (both short and long segments)
- **Diversity**: 5-10 different performers per Chopin Etude (focused approach)  
- **Consistency**: Self-correlation >0.8 on repeat ratings
- **Coverage**: All 19 perceptual dimensions with appropriate segment lengths
- **Validation**: External validation on 10-20% of segments

### Model Performance (Based on PercePiano Benchmarks)

- **Correlation**: >0.6 average across dimensions
- **Feature-Level Performance**: Different targets for different feature types
  - Technical features (timing): >0.7 correlation
  - Musical features (dynamics): >0.6 correlation  
  - Interpretive features (convincingness): >0.5 correlation
- **Generalization**: Test on held-out performers/pieces
- **Comparison**: Competitive with PercePiano results on similar features

### Key Advantages of Your Approach

✅ **Chopin Focus**: Stylistically consistent dataset vs PercePiano's multi-composer approach
✅ **Etude Selection**: Technical studies provide clear performance variation
✅ **Self-Annotation**: Consistent standards across all ratings
✅ **Expandability**: Framework designed for future growth

---

## Additional Resources

**PercePiano Dataset Access**:

- Paper: "Piano performance evaluation dataset with multilevel perceptual features" (Scientific Reports, 2024)
- Dataset: DOI 10.5281/zenodo.13269613
- Code: <https://github.com/JonghoKimSNU/PercePiano>

*Ready to begin systematic data collection with state-of-the-art methodology!* 🎼
