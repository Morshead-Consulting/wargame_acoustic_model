"""
All tunable constants in one place.
Every other module imports from here — no magic numbers in business logic.
"""

# Audio
SAMPLE_RATE: int = 16000        # Hz
MAX_ORDER: int = 12             # ISM reflection order (raise to 15 for RT60 > 0.8 s)
DEFAULT_ABSORPTION: float = 0.25

# Participant geometry
Z_SEATED: float = 1.2           # metres — mouth height when seated
Z_STANDING: float = 1.6         # metres — mouth height when standing

# MXA310 tabletop boundary array
MXA310_HEIGHT: float = 0.75        # z enforced in MicrophoneProfile
MXA310_CAPSULE_RADIUS: float = 0.05  # physical spacing of capsules from centre (m)
MXA310_CHANNELS: int = 4
MXA310_EFFECTIVE_RADIUS: float = 1.2  # 6 dB coverage radius (m)

# MXA920 / MXA910 ceiling array
MXA920_HEIGHT: float = 3.0
MXA920_BEAM_HALF_ANGLE_DEG: float = 7.5  # mid-point of 10°–15° spec range
MXA920_CHANNELS: int = 8
MXA920_EFFECTIVE_RADIUS: float = 0.8     # floor footprint radius at 3 m height

# MX412 gooseneck
MX412_HEIGHT: float = 0.90
MX412_EFFECTIVE_RADIUS: float = 0.6

# Quality thresholds
SINR_TARGET_DB: float = 15.0    # minimum isolation between zones
STOI_EXCELLENT: float = 0.85    # predicted WER < 5 %
STOI_ACCEPTABLE: float = 0.70   # dependent on DeepFilterNet

# Monte Carlo defaults
MC_DEFAULT_SIGMA: float = 0.3   # metres

# Ceiling-height DRR penalty
CEILING_PENALTY_REF_HEIGHT: float = 3.0   # no penalty at or below this height
CEILING_PENALTY_SCALE: float = 0.08       # STOI units lost per metre above baseline

# Table noise defaults
TABLE_NOISE_LOW_FREQ_HZ: float = 80.0
TABLE_NOISE_IMPULSE_RATE_HZ: float = 2.0
TABLE_NOISE_AMPLITUDE: float = 0.05
