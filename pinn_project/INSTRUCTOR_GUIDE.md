# Instructor Guide

## Multi-Method Validation Framework for Teaching Radiation Transport

---

## Overview

This guide provides instructors with detailed guidance for implementing the three classroom activities in computational physics courses. Each activity includes learning objectives, timing, setup requirements, discussion prompts, and assessment rubrics.

---

## Course Integration

### Recommended Course Types
- Computational physics (upper undergraduate)
- Nuclear instrumentation laboratory
- Radiation physics
- Introductory medical physics
- Scientific computing with applications

### Prerequisites for Students
- Python programming (intermediate level)
- Basic understanding of particle physics concepts
- Familiarity with either Monte Carlo methods OR neural networks

### Suggested Sequence
1. **Week 1**: Activity 1 (Monte Carlo Exploration) - 2 hours
2. **Week 2**: Activity 2 (NIST Comparison) - 1 hour  
3. **Week 3**: Activity 3 (PINN Investigation) - 2 hours

Can be condensed into a single 5-hour workshop or spread across multiple weeks.

---

## Activity 1: Monte Carlo Exploration

### Learning Objectives
By the end of this activity, students should be able to:
- [ ] Identify Bragg peak features in dose-depth profiles
- [ ] Explain the relationship between incident energy and peak position
- [ ] Describe the β⁻² dependence of stopping power (Bethe-Bloch)
- [ ] Articulate what physics Monte Carlo simulation captures

### Setup Requirements
- Python 3.10+ with numpy, pandas, matplotlib
- Access to training data (`pinn_training_data_v2.csv`)
- Jupyter notebook environment

### Timing (2 hours)
| Section | Time | Notes |
|---------|------|-------|
| Introduction & setup | 10 min | Brief context on MC simulation |
| Part 1-2: Data loading & profiles | 20 min | Students explore independently |
| Part 3: Peak analysis | 25 min | Quantitative investigation |
| Discussion break | 10 min | Compare observations |
| Part 4: Bethe-Bloch | 25 min | Connect to theory |
| Part 5: Extension | 20 min | Challenge activity |
| Wrap-up | 10 min | Key takeaways |

### Discussion Prompts
1. "Why does the Bragg peak exist? What happens physically as the proton slows down?"
2. "At 6 GeV, the proton traverses our 10mm detector easily. Where is it depositing most of its energy?"
3. "If we had unlimited computing power, would Monte Carlo give us the 'perfect' answer? Why or why not?"

### Expected Student Difficulties
- **Confusion about units**: MeV/mm vs MeV·cm²/g — clarify density scaling
- **Peak identification for high energies**: Students may look for peaks where protons exit the detector
- **β⁻² relationship**: Students may not recognize the non-linear relationship

### Assessment Rubric

| Criterion | Excellent (4) | Good (3) | Developing (2) | Beginning (1) |
|-----------|--------------|----------|----------------|---------------|
| Peak identification | Correctly identifies all peak positions and explains trend | Identifies peaks but incomplete explanation | Identifies some peaks, limited analysis | Cannot identify peaks |
| Bethe-Bloch connection | Explains β⁻² dependence and its physical origin | Recognizes relationship but incomplete physics | Mentions formula but no connection to data | No connection made |
| Critical thinking | Proposes thoughtful extension predictions with physics justification | Makes reasonable predictions | Attempts prediction without justification | No engagement with extension |

---

## Activity 2: Reference Data Comparison

### Learning Objectives
By the end of this activity, students should be able to:
- [ ] Validate computational results against authoritative reference data
- [ ] Compute and interpret TOPAS/NIST ratios
- [ ] Explain physics sources of systematic discrepancies
- [ ] Distinguish between computational error and missing physics

### Setup Requirements
- Same as Activity 1
- NIST PSTAR reference data (provided in notebook)

### Timing (1 hour)
| Section | Time | Notes |
|---------|------|-------|
| Introduction | 5 min | What is NIST PSTAR? |
| Part 1-3: Data comparison | 20 min | Calculate ratios |
| Part 4: Visualization | 10 min | Graphical analysis |
| Part 5: Physics explanation | 15 min | Key teaching moment |
| Discussion & wrap-up | 10 min | Validation mindset |

### Key Teaching Moment
The systematic excess at high energies is NOT an error—it reveals nuclear interactions. This is the central lesson: **disagreement between methods often reveals complementary physics rather than computational failure.**

### Discussion Prompts
1. "Before I tell you the answer: why might TOPAS predict higher stopping power than NIST at high energies?"
2. "If you were validating a new simulation code, what would you conclude from 15% disagreement with NIST at 1 GeV?"
3. "What assumption underlies NIST PSTAR data? When does this assumption break down?"

### Common Misconceptions
- "The simulation is wrong because it doesn't match NIST" — Address by explaining what NIST actually reports
- "We should adjust our simulation to match NIST" — Clarify that TOPAS includes MORE physics
- "Agreement means the simulation is correct" — Discuss limitations of validation

### Assessment Rubric

| Criterion | Excellent (4) | Good (3) | Developing (2) | Beginning (1) |
|-----------|--------------|----------|----------------|---------------|
| Ratio calculation | Correctly computes all ratios and identifies trends | Minor computation errors but correct interpretation | Computes some ratios, incomplete analysis | Cannot compute ratios |
| Physics explanation | Articulates nuclear interaction contribution clearly | Mentions nuclear effects but incomplete | Vague physics explanation | No physics explanation |
| Validation mindset | Demonstrates understanding that disagreement reveals physics | Recognizes disagreement is not always error | Unsure how to interpret disagreement | Assumes all disagreement is error |

---

## Activity 3: PINN Investigation

### Learning Objectives
By the end of this activity, students should be able to:
- [ ] Compare baseline vs physics-constrained ML model performance
- [ ] Interpret ablation study results
- [ ] Explain why some physics constraints help and others hurt
- [ ] Articulate the lesson: "Adding physics requires validation"

### Setup Requirements
- TensorFlow 2.15+
- Pre-trained models (`baseline_nn.keras`, `advanced_pinn.keras`)
- Ablation study results (`ablation_study.csv`)

### Timing (2 hours)
| Section | Time | Notes |
|---------|------|-------|
| Introduction to PINNs | 15 min | Conceptual overview |
| Part 1-3: Model comparison | 25 min | Hands-on evaluation |
| Discussion break | 10 min | Process findings |
| Part 4: Ablation study | 30 min | Critical analysis |
| Part 5: Surprising lesson | 20 min | Key teaching moment |
| Part 6: Student investigation | 15 min | Propose new constraints |
| Wrap-up | 5 min | Synthesis |

### Key Teaching Moment
The gradient constraint (based on Bethe-Bloch) **hurts** performance. This counterintuitive result teaches that:
1. Even well-established physics formulas have domains of validity
2. Physics constraints must themselves be validated
3. "Adding physics" is not automatically good

### Discussion Prompts
1. "Predict which constraint will help most before looking at the ablation results. Note your prediction."
2. "The gradient constraint is based on a Nobel Prize-winning formula (Bethe). How can adding it make predictions worse?"
3. "In what other ML applications might 'adding domain knowledge' backfire?"
4. "How would you decide whether to include a new physics constraint?"

### Advanced Discussion (for graduate students)
- Bias-variance tradeoff interpretation
- When to use hard vs soft constraints
- Multi-fidelity learning approaches
- Transfer learning with physics constraints

### Assessment Rubric

| Criterion | Excellent (4) | Good (3) | Developing (2) | Beginning (1) |
|-----------|--------------|----------|----------------|---------------|
| Model comparison | Correctly evaluates both models and quantifies improvement | Evaluates models but incomplete analysis | Limited comparison, some errors | Cannot compare models |
| Ablation interpretation | Explains why gradient constraint hurts with physics reasoning | Identifies harmful constraint but incomplete explanation | Recognizes some constraints help/hurt | Cannot interpret ablation |
| Critical thinking | Proposes thoughtful new constraint with validation plan | Proposes reasonable constraint | Constraint proposal without analysis | No engagement with extension |
| Key lesson | Articulates "physics constraints require validation" clearly | Demonstrates partial understanding | Vague understanding | Does not grasp key lesson |

---

## Overall Assessment

### Portfolio Assessment
Have students compile their work across all three activities into a validation portfolio that includes:
1. Key figures with annotations
2. Written responses to questions
3. Reflection on the validation mindset

### Summative Assessment Questions

1. **Conceptual**: "You run two different simulations of the same physics problem and get different answers. Describe a systematic approach to determine which (if either) is correct."

2. **Application**: "A colleague adds a physics constraint to their ML model and sees 5% improvement on validation data. What questions would you ask before concluding this is beneficial?"

3. **Synthesis**: "Compare and contrast the validation approaches used in Activities 1, 2, and 3. What common principles emerge?"

---

## Frequently Asked Questions

**Q: What if students don't have access to GPUs?**
A: All models can run on CPU. Training from scratch takes longer (~1 hour) but works fine. Pre-trained models provide instant results.

**Q: Can this be adapted for high school students?**
A: Activity 1 is accessible with simplified explanations. Activities 2-3 require calculus and programming background.

**Q: What if TOPAS/simulation setup is beyond scope?**
A: The framework works entirely with pre-computed data. Students don't need simulation access.

**Q: How do I assess whether students achieved the validation mindset?**
A: Look for evidence that students:
- Question computational results rather than accepting blindly
- Seek multiple independent methods for validation
- Interpret disagreement as information, not just error
- Recognize limitations of any single approach

---

## Additional Resources

- [NIST PSTAR Database](https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html)
- [OpenTOPAS Documentation](https://opentopas.readthedocs.io/)
- [Geant4 Physics Lists](https://geant4.web.cern.ch/collaboration/working_groups/physics)
- [Physics-Informed Neural Networks Review](https://doi.org/10.1038/s42254-021-00314-5)

---

## Contact

For questions about implementation, please contact:
- **Yash Varshney** - yash.gurukul12@gmail.com
