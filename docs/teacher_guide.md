# Teacher guide

This guide helps you use **kidney-sim** in a secondary-school classroom
(biology, physics, chemistry, maths or technology; ages ~15–18).
Each level is planned as **one session of about 60 minutes**. You can stop after
any level: each one is complete on its own and ends with "what is missing?",
which leads into the next.

## Before you start

- Students need a Google account to use Colab (or a local Python install).
- No programming experience is required for Levels 0–3: students run cells and move sliders.
  Students who know a little Python can edit the Euler loop in Level 0 and the parameters in later levels.
- Maths needed: rates of change (no calculus required: Euler's method is explained as "small steps"),
  proportions, reading graphs.

## Overview

| Level | Main idea | Subjects | Key vocabulary |
|---|---|---|---|
| 0 | Mass balance: in − out = change | maths, biology | clearance, steady state, time constant |
| 1 | Filtration is driven by pressure | physics, biology | hydrostatic and oncotic pressure, resistance, autoregulation |
| 2 | Membrane transport recovers useful substances | chemistry, biology | transporter, saturation, exchanger, osmosis, hormone |
| 3 | Countercurrent multiplication | physics, biology | gradient, countercurrent, osmolarity |
| 4 | Homeostasis via negative feedback | biology | ADH, aldosterone, thirst, feedback loop |
| 5 | Engineering an artificial kidney | chemistry, technology | diffusion, dialysis, ion exchange, breakthrough |

## The video (notebook 06)

The 80-second video (`docs/video/`, Spanish and English) shows the whole journey from blood to urine.
Two good ways to use it:

- **Before Level 0**, as a hook: "What did you see? Which numbers surprised you?"
- **After Level 2**, as a summary. Students then render a new video for a patient
  (no ADH, diabetes) and explain what changes.

## Suggested session plan (each level)

1. **Hook (5 min).** A real-life question:
   - L0: "Why do people with kidney failure need dialysis several times a week?"
   - L1: "What happens to your kidneys if you lose a lot of blood?"
   - L2: "Why is there sugar in the urine of someone with diabetes?"
   - L3: "How can a kangaroo rat live without drinking?"
   - L4: "Why do you pee more when you drink lots of water, but your blood stays the same?"
   - L5: "How does a dialysis machine work?"
2. **Predict (5 min).** Students write a prediction before running the code.
3. **Explore (25 min).** Run the notebook; read the equation cell together; move the sliders.
4. **Explain (15 min).** Answer the questions at the end of the notebook in small groups.
5. **Connect (10 min).** "What is missing?" Which simplification would you remove next?

## Common misconceptions to address

| Misconception | What really happens | Where the model shows it |
|---|---|---|
| "The kidney makes urea." | The **liver** makes urea; the kidney excretes it. | Level 0: $G$ is the liver, $K$ is the kidney |
| "The kidney cleans blood like a sieve that only lets waste through." | It filters **everything small** (180 L/day) and then takes back what is useful. | Levels 1–2 |
| "Ion exchange is how the kidney removes waste." | Waste is removed mainly by filtration. Exchangers do exist (Na⁺/H⁺, OAT1) and resin-type ion exchange is used in dialysis sorbents. | Level 2 (OAT secretion), Level 5 |
| "Pressure does not matter inside the body." | Glomerular filtration is driven entirely by blood pressure. | Level 1 autoregulation plot |
| "More water in = more water in the blood." | ADH and thirst keep plasma osmolarity almost constant. | Level 4 `drink_5_litres` |
| "Urine is always the same." | Urine can be 4× more concentrated than blood or 6× more dilute. | Level 2 ADH slider |

## Assessment ideas

- **Short answer.** Using $C_{ss} = G/K$, explain why creatinine is used to estimate kidney function.
- **Graph reading.** On the glucose titration curve, mark the threshold and explain *saturation*.
- **Feedback diagram.** Draw the ADH loop: stimulus → sensor → hormone → effector → response.
- **Mini-project (2–3 sessions).** Remove one simplification listed in the README
  (e.g. add potassium, or make ADH respond with a delay), and write a one-page report
  with the new equation, a figure and a comparison with textbook values.
- **Engineering challenge.** In Level 5, find the cheapest settings (shortest session,
  lowest blood flow) that still give $Kt/V \ge 1.2$.

## Cross-curricular links

- **Maths:** exponential approach to a steady state; Euler's method; Michaelis–Menten saturation.
- **Physics:** pressure, flow and resistance (Ohm's law analogy); diffusion; countercurrent exchangers
  (also found in fish gills and in industrial heat exchangers).
- **Chemistry:** concentration, osmolarity, equilibrium, the Langmuir isotherm, ion exchange resins (water softeners).
- **Biology:** nephron structure, hormones, homeostasis, diabetes, kidney disease.
- **Technology / ethics:** dialysis, transplantation, organ donation.

## Safety note

This is an **educational model**. Parameters are textbook averages; it must not be used
for medical decisions.
