"""
Reference values for a healthy adult (about 70 kg).

These numbers are used to set up the models and to check that the
results look like a real kidney. They are typical textbook values;
real people vary a lot around them.

Units used everywhere in this package
-------------------------------------
pressure        mmHg
flow            L/min   (we print mL/min for readability)
concentration   mmol/L  (= mOsm/L for a solute that does not split into ions)
amount          mmol
time            min     (body-level models report days or hours)
"""

# ---- Body -------------------------------------------------------------
BODY_WATER_L = 42.0          # total body water (60 % of 70 kg)
PLASMA_NA = 140.0            # mmol/L
PLASMA_UREA = 5.0            # mmol/L   (normal 2.5 - 7.5)
PLASMA_GLUCOSE = 5.0         # mmol/L   (fasting, normal 4 - 6)
PLASMA_CREATININE = 0.07     # mmol/L   (= 70 umol/L)
PLASMA_OSM = 290.0           # mOsm/L
PLASMA_PROTEIN = 7.0         # g/dL     (mostly albumin)

UREA_GENERATION = 0.28       # mmol/min (~400 mmol/day, made by the LIVER)
CREATININE_GENERATION = 0.0087  # mmol/min (~12.5 mmol/day, made by muscles)

# ---- Kidney blood flow and filtration ---------------------------------
MAP = 100.0                  # mean arterial pressure, mmHg
P_RENAL_VEIN = 4.0           # mmHg
RENAL_BLOOD_FLOW = 1.1       # L/min (about 20 % of cardiac output)
HEMATOCRIT = 0.45            # fraction of blood that is red cells
P_BOWMAN = 15.0              # pressure in Bowman's space, mmHg
P_GLOMERULUS = 60.0          # glomerular capillary pressure, mmHg

GFR_NORMAL = 0.125           # L/min  (= 125 mL/min = 180 L/day)
URINE_FLOW_NORMAL = 1.0e-3   # L/min  (~1.4 L/day)

# ---- Normal ranges used to draw the green "healthy" bands ----------------
NORMAL_RANGES = {
    "gfr_ml_min": (90, 140),
    "urea_mmol_l": (2.5, 7.5),
    "creatinine_umol_l": (45, 110),
    "urine_flow_ml_min": (0.4, 2.0),
    "urine_osm": (50, 1200),
    "plasma_osm": (280, 298),
}
