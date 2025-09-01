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

## Recommended Repertoire

### Chopin Selection (Perceptual Diversity)

```
Technical/Virtuosic:
- Etude Op.10 No.1 (arpeggios - articulation focus)
- Etude Op.10 No.4 (velocity - timing precision)
- Etude Op.25 No.6 (thirds - dynamic control)

Lyrical/Expressive:
- Nocturne Op.9 No.2 (rubato - timing flexibility)
- Nocturne Op.27 No.2 (pedaling - sustain effects)
- Ballade No.1 Op.23 (narrative - musical expression)

Rhythmic/Dance:
- Waltz Op.64 No.2 (tempo stability)
- Polonaise Op.53 (articulation contrasts)
- Mazurka Op.17 No.4 (folk characteristics)
```

### Liszt Selection (Extended Techniques)

```
Transcendental Difficulty:
- Hungarian Rhapsody No.2 (dramatic contrasts)
- Transcendental Etude No.10 "Allegro" (velocity/power)
- La Campanella (precision/lightness)

Lyrical Works:
- Liebestraum No.3 (legato/pedaling)
- Consolation No.3 (tone quality)
- Un Sospiro (voicing/balance)
```

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

```python
# Based on your preprocessing pipeline:
segment_length = 3.0  # seconds
overlap = 0.5        # 50% overlap for more training data

# Generate segments per performance:
# - 2-3 minute piece → ~8-12 segments
# - Provides multiple training examples per performance
```

### 3. Quality Control Checklist

- [ ] Audio plays without artifacts
- [ ] Consistent volume level across segments
- [ ] No clipping or distortion
- [ ] Clear piano sound (not muffled/distant)
- [ ] Minimal background noise

## Labeling Strategy

### 1. Perceptual Dimensions (19 total)

Based on PercePiano research:

```
Timing: Stable ←→ Unstable
Articulation: Short/Soft ←→ Long/Hard  
Pedal: Dry/Clean ←→ Wet/Blurred
Timbre: Even/Shallow/Bright/Soft ←→ Colorful/Rich/Dark/Loud
Dynamics: Mellow/Small ←→ Raw/Large Range
Musical: Fast/Flat/Unbalanced/Pure ←→ Slow/Spacious/Balanced/Expressive
Emotion: Pleasant/Low Energy/Honest ←→ Dark/High Energy/Imaginative
Interpretation: Poor ←→ Convincing
```

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

### 3. Consistency Checks

- Rate same segment twice (separated by time)
- Target consistency: >0.8 correlation with your previous ratings
- Include some PercePiano segments as calibration anchors

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

**Solution**: Rate in 30-minute sessions, max 20 segments per session

### Challenge: Consistency Across Sessions  

**Solution**: Include 2-3 "anchor" segments per session for calibration

### Challenge: Copyright Issues

**Solution**: Focus on competition recordings, IMSLP, and personal recordings

### Challenge: Performer Bias

**Solution**: Balance skill levels, include student + professional recordings

## Success Metrics

### Dataset Quality

- **Size**: 300+ labeled segments
- **Diversity**: 10+ different performers per major piece  
- **Consistency**: Self-correlation >0.8 on repeat ratings
- **Coverage**: All 19 perceptual dimensions adequately represented

### Model Performance

- **Correlation**: >0.6 average across dimensions (vs 0.5+ on synthetic)
- **Generalization**: Test on held-out performers/pieces
- **Comparison**: Competitive with PercePiano baseline results

---

*Ready to begin systematic data collection for robust Chopin/Liszt performance analysis!* 🎼
