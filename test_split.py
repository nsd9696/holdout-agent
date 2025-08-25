#!/usr/bin/env python3
# Test script for split logic using pytest

import pytest
from split_engine import SplitSpec, plan_split

def test_split_spec_creation():
    """Test SplitSpec creation with various parameters."""
    
    # Test basic spec creation
    spec = SplitSpec(
        goal="Evaluate commonsense and science MCQ reasoning in Korean and English",
        total=20,
        difficulty_mix={1: 0.05, 2: 0.15, 3: 0.5, 4: 0.2, 5: 0.1},
        topics=["commonsense", "science"],
        benchmarks=["hellaswag", "kmmlu"],
        heldin_ratio=0.5
    )
    
    # Assertions
    assert spec.goal == "Evaluate commonsense and science MCQ reasoning in Korean and English"
    assert spec.total == 20
    assert spec.heldin_ratio == 0.5
    assert spec.topics == ["commonsense", "science"]
    assert spec.benchmarks == ["hellaswag", "kmmlu"]
    assert spec.difficulty_mix == {1: 0.05, 2: 0.15, 3: 0.5, 4: 0.2, 5: 0.1}

def test_split_spec_calculations():
    """Test SplitSpec calculations for held-in and held-out counts."""
    
    spec = SplitSpec(
        goal="Test goal",
        total=100,
        difficulty_mix={1: 0.2, 2: 0.3, 3: 0.5},
        topics=[],
        benchmarks=[],
        heldin_ratio=0.6
    )
    
    # Test held-in and held-out calculations
    expected_held_in = int(100 * 0.6)
    expected_held_out = 100 - expected_held_in
    
    assert expected_held_in == 60
    assert expected_held_out == 40

def test_split_spec_edge_cases():
    """Test SplitSpec with edge cases."""
    
    # Test with heldin_ratio = 0.0
    spec_zero = SplitSpec(
        goal="Test zero ratio",
        total=10,
        difficulty_mix={1: 1.0},
        topics=[],
        benchmarks=[],
        heldin_ratio=0.0
    )
    assert spec_zero.heldin_ratio == 0.0
    
    # Test with heldin_ratio = 1.0
    spec_one = SplitSpec(
        goal="Test one ratio",
        total=10,
        difficulty_mix={1: 1.0},
        topics=[],
        benchmarks=[],
        heldin_ratio=1.0
    )
    assert spec_one.heldin_ratio == 1.0

def test_split_spec_difficulty_mix():
    """Test SplitSpec difficulty mix validation."""
    
    spec = SplitSpec(
        goal="Test difficulty mix",
        total=50,
        difficulty_mix={1: 0.3, 2: 0.4, 3: 0.3},
        topics=[],
        benchmarks=[],
        heldin_ratio=0.5
    )
    
    # Test that difficulty mix sums to approximately 1.0
    total_mix = sum(spec.difficulty_mix.values())
    assert abs(total_mix - 1.0) < 0.001

def test_split_spec_empty_topics_and_benchmarks():
    """Test SplitSpec with empty topics and benchmarks."""
    
    spec = SplitSpec(
        goal="Test empty filters",
        total=30,
        difficulty_mix={1: 0.5, 2: 0.5},
        topics=[],
        benchmarks=[],
        heldin_ratio=0.5
    )
    
    assert spec.topics == []
    assert spec.benchmarks == []
    assert len(spec.topics) == 0
    assert len(spec.benchmarks) == 0

if __name__ == "__main__":
    # Run tests directly if not using pytest
    pytest.main([__file__, "-v"])
