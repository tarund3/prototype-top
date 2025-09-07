# Qualitative Examples

## Example 1: Better Token Ranking

**Input**: "The quick brown fox"

**NTP Prediction**: "jumps" (next token only)

**TOP Prediction**: 
- "jumps" (rank 1, distance 1)
- "runs" (rank 2, distance 3) 
- "walks" (rank 3, distance 5)
- "sleeps" (rank 4, distance 8)

**Analysis**: TOP provides richer context about future tokens, not just the immediate next one.

## Example 2: Improved Coherence

**Input**: "In the beginning"

**NTP**: "of" → "time" → "there" → "was" → "darkness"

**TOP**: "of" → "the" → "universe" → "there" → "was" → "nothing"

**Analysis**: TOP's ranking helps maintain better long-term coherence by understanding token relationships.

## Example 3: Domain-Specific Knowledge

**Input**: "The mitochondria is"

**NTP**: "the" → "powerhouse" → "of" → "the" → "cell"

**TOP**: "the" → "powerhouse" → "of" → "the" → "cell" → "and" → "produces" → "ATP"

**Analysis**: TOP's ranking captures domain-specific knowledge better by understanding scientific terminology patterns.
    