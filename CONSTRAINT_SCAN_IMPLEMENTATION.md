# Constraint Scan Fallback for TS Optimization - Implementation Plan

## Overview
When all TS guesses fail (freq or NMD failures) and no other options exist, use a Gaussian constraint scan along the reaction coordinate to find the approximate TS, then optimize it.

## Implementation Components

### 1. Detection Logic (arc/scheduler.py)
**Location**: `switch_ts()` method, around line 3006
**Trigger**: When `ts_guesses_exhausted` is True and no viable guesses remain

```python
# In switch_ts() method, after line 3010:
elif not self.species_dict[label].ts_guess_priority:
    # No viable TS guess selected (e.g., only a failed user guess). Fall back to automated TS guessing.
    if self.species_dict[label].tsg_spawned and self.species_dict[label].ts_guesses_exhausted:
        # NEW: Try constraint scan as last resort
        if not getattr(self.species_dict[label], 'constraint_scan_attempted', False):
            logger.info(f'All TS guesses for {label} exhausted. Attempting constraint scan fallback...')
            self.species_dict[label].constraint_scan_attempted = True
            success = self.run_constraint_scan_for_ts(label)
            if success:
                return
        logger.info(f'All TS guesses for {label} were already generated and exhausted; '
                    f'not spawning another round.')
        return
```

### 2. Reaction Coordinate Identification

**New Method**: `identify_reaction_coordinate(label: str) -> Optional[Dict]`

For H-abstraction reactions:
- Parse reactants and products
- Identify breaking bond (D-H)
- Identify forming bond (A-H)
- Return dict with atom indices

```python
def identify_reaction_coordinate(self, label: str) -> Optional[Dict]:
    """
    Identify the reaction coordinate for constraint scan.
    Currently supports H-abstraction reactions.
    
    Returns:
        Dict with keys: 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'reaction_type'
        None if cannot identify
    """
    if label not in self.species_dict or not self.species_dict[label].is_ts:
        return None
    
    rxn_index = self.species_dict[label].rxn_index
    if rxn_index not in self.rxn_dict:
        return None
    
    rxn = self.rxn_dict[rxn_index]
    
    # Check if this is an H-abstraction
    if rxn.family and 'H_Abstraction' in rxn.family.label:
        return self._identify_h_abstraction_coordinate(rxn)
    
    # TODO: Add support for other reaction families
    return None

def _identify_h_abstraction_coordinate(self, rxn) -> Optional[Dict]:
    """
    For H-abstraction: R-H + X· → R· + H-X
    Identify: donor (R), hydrogen (H), acceptor (X)
    """
    # Use atom mapping from RMG to identify which atoms participate
    # in the bond breaking/forming
    
    # This requires accessing the reaction's atom mapping
    # which is typically available in rxn.pairs or similar
    
    # Placeholder implementation:
    # You'll need to implement logic to:
    # 1. Find bonds that break (R-H)
    # 2. Find bonds that form (H-X)
    # 3. Map these to atom indices in the TS geometry
    
    return {
        'donor_idx': None,  # To be determined from mapping
        'hydrogen_idx': None,
        'acceptor_idx': None,
        'reaction_type': 'H_Abstraction'
    }
```

### 3. Constraint Scan Job Creation

**New Method**: `run_constraint_scan_for_ts(label: str) -> bool`

```python
def run_constraint_scan_for_ts(self, label: str) -> bool:
    """
    Run a constraint scan along the reaction coordinate to find TS guess.
    
    Steps:
    1. Identify reaction coordinate (D-H-A atoms)
    2. Create constraint scan job
    3. Parse results to find maximum
    4. Create new TS guess from maximum
    5. Optimize as TS
    """
    # Identify reaction coordinate
    coord_info = self.identify_reaction_coordinate(label)
    if coord_info is None:
        logger.warning(f'Could not identify reaction coordinate for {label}. '
                      f'Constraint scan fallback not available.')
        return False
    
    logger.info(f'Identified reaction coordinate for {label}: '
                f'D={coord_info["donor_idx"]}, H={coord_info["hydrogen_idx"]}, '
                f'A={coord_info["acceptor_idx"]}')
    
    # Create initial geometry from reactants
    initial_xyz = self._create_reactant_complex_geometry(label)
    if initial_xyz is None:
        logger.error(f'Could not create reactant complex geometry for {label}')
        return False
    
    # Run constraint scan job
    scan_job_name = f'constraint_scan_{label}'
    self.run_job(
        label=label,
        xyz=initial_xyz,
        level_of_theory=self.ts_guess_level,
        job_type='constraint_scan',
        coord_info=coord_info,
        scan_nsteps=25,
        scan_stepsize=0.05,
    )
    
    return True

def _create_reactant_complex_geometry(self, label: str) -> Optional[dict]:
    """
    Create an initial geometry with reactants in close proximity.
    This is the starting point for the constraint scan.
    """
    # Use reactant geometries and place them close together
    # Orient them appropriately for the reaction
    # This is non-trivial and may require molecular mechanics or heuristics
    pass
```

### 4. Gaussian Adapter Extension

**File**: `arc/job/adapters/gaussian.py`

Add support for 'constraint_scan' job type:

```python
# In write_input_file method, add new job_type handling:

elif self.job_type == 'constraint_scan':
    # Constraint scan for TS search
    coord_info = self.args.get('coord_info', {})
    donor = coord_info.get('donor_idx')
    hydrogen = coord_info.get('hydrogen_idx')
    acceptor = coord_info.get('acceptor_idx')
    nsteps = self.args.get('scan_nsteps', 25)
    stepsize = self.args.get('scan_stepsize', 0.05)
    
    if donor is None or hydrogen is None or acceptor is None:
        raise ValueError('Constraint scan requires donor, hydrogen, and acceptor indices')
    
    # Use ModRedundant for constraint scan
    input_dict['job_type_1'] = 'opt=modredundant'
    
    # Add scan constraints
    # Option A: Scan B1 (D-H distance), freeze B2 (A-H distance)
    input_dict['scan'] = f'\n\nB {donor} {hydrogen} F\n'  # Freeze A-H
    input_dict['scan'] += f'B {donor} {hydrogen} S {nsteps} {stepsize:.3f}\n'  # Scan D-H
    
    # Alternative Option B: Scan difference D = B1 - B2
    # This is more complex and requires custom coordinate definition
```

### 5. Parse Constraint Scan Results

**File**: `arc/parser/adapters/gaussian.py`

Extend parsing to extract scan energies and geometries:

```python
# Enhance parse_1d_scan_energies to handle constraint scans
# Add method to extract geometry at maximum energy point
```

**New Method in scheduler.py**:

```python
def process_constraint_scan_results(self, label: str, scan_job):
    """
    Process constraint scan results and create TS guess from maximum.
    """
    # Parse scan energies
    energies, _ = parser.parse_1d_scan_energies(scan_job.local_path_to_output_file)
    
    if energies is None:
        logger.error(f'Could not parse constraint scan energies for {label}')
        return False
    
    # Find maximum energy point
    max_idx = energies.index(max(energies))
    logger.info(f'Maximum energy found at scan point {max_idx} '
                f'with relative energy {max(energies):.2f} kJ/mol')
    
    # Extract geometry at maximum
    scan_geometries = parser.parse_scan_conformers(scan_job.local_path_to_output_file)
    if scan_geometries is None:
        logger.error(f'Could not parse scan geometries for {label}')
        return False
    
    max_geometry = scan_geometries.iloc[max_idx]['xyz']
    
    # Create new TS guess from this geometry
    new_tsg = TSGuess(
        method='constraint_scan',
        initial_xyz=max_geometry,
        index=len(self.species_dict[label].ts_guesses),
        success=True,
        energy=None,  # Will be calculated during opt
    )
    
    self.species_dict[label].ts_guesses.append(new_tsg)
    self.species_dict[label].chosen_ts = new_tsg.index
    self.species_dict[label].ts_guesses_exhausted = False
    
    # Now run TS optimization from this guess
    logger.info(f'Created TS guess from constraint scan maximum. '
                f'Starting TS optimization for {label}...')
    self.run_opt_job(label, fine=self.fine_only)
    
    return True
```

### 6. Integration Points

**Key integration points in scheduler.py**:

1. **End of `check_freq_job()`** - If freq fails and no more guesses:
   - Set flag to trigger constraint scan
   
2. **In `switch_ts()`** - Check for constraint scan flag:
   - If set and not already attempted, run constraint scan

3. **New job completion handler** - For constraint_scan jobs:
   - Parse results
   - Extract maximum
   - Create TS guess
   - Trigger optimization

### 7. Testing Strategy

1. **Unit tests**: Test reaction coordinate identification for known H-abstractions
2. **Integration test**: Mock constraint scan with known maximum
3. **End-to-end test**: Run on actual failed TS case

### 8. Limitations and Future Work

**Current scope** (MVP):
- H-abstraction reactions only
- Simple distance scan (Option A)
- Gaussian only

**Future enhancements**:
- Other reaction families (e.g., additions, eliminations)
- Scan difference coordinate (Option B)
- Support for other QM codes
- Automatic reactant complex generation
- Multi-dimensional scans for complex reactions

## Implementation Order

1. ✅ Add detection logic in `switch_ts()` - COMPLETED
2. ✅ Create `identify_reaction_coordinate()` stub - COMPLETED (stub exists, needs RMG mapping)
3. ✅ Add Gaussian constraint scan job type - COMPLETED
4. ✅ Implement scan result parsing - COMPLETED (using existing parser)
5. ✅ Connect constraint scan to TS optimization pipeline - COMPLETED
6. ⏳ Test with known H-abstraction reaction - PENDING (needs coordinate identification)
7. ⏳ Refine and debug - PENDING
8. ✅ Add tests - PARTIALLY COMPLETED (unit tests added, integration tests pending)

## Code Organization

```
arc/
├── scheduler.py (main coordination logic)
├── job/
│   └── adapters/
│       └── gaussian.py (constraint scan job creation)
├── parser/
│   └── adapters/
│       └── gaussian.py (scan result parsing)
└── reaction/
    └── coordinate.py (NEW: reaction coordinate identification)
```

## Notes

- This is a last-resort method, used only when all other TS guess methods fail
- Requires accurate reaction coordinate identification
- Computational cost: scan + optimization (significant, but worth it if TS is found)
- Success rate depends on reaction type and initial geometry quality
