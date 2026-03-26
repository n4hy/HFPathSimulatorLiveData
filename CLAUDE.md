# HF Path Simulator - Claude Code Instructions

## User Preferences (n4hy)

### No Fudge Factors - Ever
- Never apply arbitrary scaling factors, magic numbers, or "fudge factors" to make results look correct
- If output doesn't match expected values, debug systematically to find the root cause
- If it takes all night, debug it properly rather than applying bandaids
- Every constant must have a physical or mathematical justification

### Require Understanding Over Guessing
- Do not guess at solutions - understand the problem first
- Systematically isolate issues using stage-by-stage analysis
- Use test values and power measurements to trace signal flow
- Reject any "fix" that cannot be explained from first principles

### Signal Processing Standards
- All signal processing must be in optimized compiled code (CUDA/C++), not Python
- Power should be preserved through processing chains (ratio ~1.0 for unity gain)
- Use proper dimensional analysis and verify physics constraints
