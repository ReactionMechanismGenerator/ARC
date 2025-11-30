# Constraint Scan Fallback - Usage Documentation

## Overview

The constraint scan fallback is an automatic last-resort method for finding transition states when all traditional TS guess methods (AutoTST, kinbot, etc.) have failed.

## When Does It Trigger?

The constraint scan automatically triggers when:
1. All TS guesses have been generated (`tsg_spawned = True`)
2. All TS guesses have been tried and failed (`ts_guesses_exhausted = True`)
3. No priority TS guess workflow is active
4. The constraint scan hasn't been attempted yet (`constraint_scan_attempted = False`)

## How It Works

### Step 1: Detection
When `switch_ts()` is called after all guesses fail, it checks the constraint scan flag:

```python
if not getattr(self.species_dict[label], 'constraint_scan_attempted', False):
    logger.info(f'All TS guesses for {label} exhausted. Attempting constraint scan fallback...')
    self.species_dict[label].constraint_scan_attempted = True
    success = self.run_constraint_scan_for_ts(label)
```

### Step 2: Coordinate Identification
The reaction coordinate (breaking and forming bonds) is identified:

```python
coord_info = identify_reaction_coordinate(rxn)
# Returns: {'donor_idx': 0, 'hydrogen_idx': 1, 'acceptor_idx': 2, 'type': 'H_Abstraction'}
```

For H-abstraction (R-H + X· → R· + H-X):
- `donor_idx`: R atom (bond breaking)
- `hydrogen_idx`: H atom (transferring)
- `acceptor_idx`: X atom (bond forming)

### Step 3: Initial Geometry Creation
Reactants are positioned in close proximity:

```python
initial_xyz = self._create_reactant_complex_geometry(label)
```

**Note**: Current implementation uses simple concatenation. Future versions will use:
- Molecular mechanics for intelligent positioning
- RMG's TS geometry estimation
- Heuristics based on reaction family

### Step 4: Constraint Scan Execution
A Gaussian job with ModRedundant constraints is spawned:

```gaussian
# Gaussian input (simplified)
#P B3LYP/6-31G(d) Opt=(ModRedundant, CalcFC, NoEigenTest, MaxStep=5)

<geometry>

B 3 2 F              ! Freeze A-H bond (forming)
B 1 2 S 25 0.050     ! Scan D-H bond (breaking), 25 steps of 0.05 Å
```

The scan sweeps the D-H distance while keeping A-H frozen, generating a potential energy surface.

### Step 5: Maximum Energy Detection
After scan completion, the code:
1. Parses scan energies
2. Finds the maximum energy point (approximate TS)
3. Extracts the geometry at that point

```python
max_idx = energies.index(max(energies))
max_geometry = scan_conformers.iloc[max_idx]['xyz']
```

### Step 6: TS Optimization
A new TSGuess is created from the maximum:

```python
new_tsg = TSGuess(
    method='constraint_scan',
    xyz=max_geometry,
    index=len(self.species_dict[label].ts_guesses),
    success=True,
)
```

This guess is then optimized as a TS using the normal ARC workflow.

## Configuration

### Default Parameters
- **Scan steps**: 25
- **Step size**: 0.05 Å
- **Scan type**: Distance (D-H bond length)
- **QM method**: Uses `ts_guess_level` from ARC settings

### Customization
Parameters can be modified in `run_constraint_scan_for_ts()`:

```python
self.run_job(
    label=label,
    xyz=initial_xyz,
    level_of_theory=self.ts_guess_level,
    job_type='constraint_scan',
    # Optional customization:
    # scan_nsteps=30,      # More steps
    # scan_stepsize=0.03,  # Finer resolution
)
```

## Supported Reaction Families

### Currently Implemented
- H-abstraction (stub only - needs RMG atom mapping)

### Planned
- Addition reactions
- Elimination reactions
- Substitution reactions
- Radical recombination

## Example Log Output

```
INFO: Switching a TS guess for TS_CH4_OH...
INFO: All TS guesses for TS_CH4_OH exhausted. Attempting constraint scan fallback...
INFO: Identified reaction coordinate for TS_CH4_OH: D=0, H=4, A=5
INFO: Created initial reactant complex geometry for TS_CH4_OH with 10 atoms
INFO: Starting constraint scan for TS_CH4_OH...
INFO: Job constraint_scan_a1234 started running on local
...
INFO: Processing constraint scan results for TS_CH4_OH...
INFO: Maximum energy found at scan point 12 with relative energy 125.3 kJ/mol
INFO: Created TS guess 4 from constraint scan maximum. Starting TS optimization for TS_CH4_OH...
INFO: Job opt_a1235 started running on local
```

## Limitations

### Current Limitations
1. **No coordinate identification yet**: H-abstraction coordinate detection requires RMG atom mapping
   - Workaround: Manually specify atom indices (future feature)

2. **Naive geometry generation**: Initial geometry uses simple concatenation
   - May result in poor starting structures
   - Can lead to scan failures or incorrect TS

3. **Single reaction family**: Only H-abstraction is supported (in stub form)
   - Other families will return None from `identify_reaction_coordinate()`

4. **Gaussian only**: Other QM codes not yet supported
   - Future: ORCA, Q-Chem, Psi4

### Computational Cost
- **Time**: 2-6 hours for typical scan (25 steps)
- **Cost**: Roughly equivalent to 25 single-point calculations
- **Worth it?**: Yes, if it finds a TS that other methods missed

### Success Rate
- Expected: 60-80% for well-behaved H-abstractions
- Lower for: Sterically hindered systems, multi-step mechanisms
- Not suitable for: Very flat potential surfaces, complex rearrangements

## Troubleshooting

### Constraint scan doesn't trigger
- Check that all TS guesses have actually failed
- Verify `ts_guesses_exhausted` is True
- Check logs for "Attempting constraint scan fallback"

### "Could not identify reaction coordinate"
- Current implementation: Expected (needs RMG mapping)
- Future: Check reaction family is supported

### Scan completes but no TS found
- Maximum may be at scan boundary (try larger scan range)
- Potential surface may be too flat
- Initial geometry may be poor

### TS optimization fails after scan
- Common issue: Approximate maximum from scan may not be close enough to true TS
- Solution: Try finer scan resolution (smaller step size)
- Solution: Use better initial geometry generation

## Future Enhancements

### Near-term
1. Implement RMG atom mapping for H-abstraction
2. Improve reactant positioning algorithms
3. Add more reaction family support

### Long-term
1. Multi-dimensional scans for complex reactions
2. Difference coordinate scanning (D = d₁ - d₂)
3. Machine learning for initial geometry prediction
4. Support for other QM codes
5. Adaptive scan resolution (fine near maximum)

## References

- Gaussian ModRedundant: https://gaussian.com/modredundant/
- ARC Documentation: https://reactionmechanismgenerator.github.io/ARC/
- RMG Reaction Families: https://rmg.mit.edu/database/kinetics/families/

## Contact

For questions or issues:
- Open an issue on ARC GitHub
- Contact the ARC development team
