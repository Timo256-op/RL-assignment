# RL_assigment/env/teacher_alloc_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from itertools import product

class TeacherAllocEnv(gym.Env):
    """
    Teacher allocation environment.
    Decision epoch: weekly.
    - C classes
    - J total teachers (identical for simplicity)
    Actions: discrete allocation patterns (encoded as indices).
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self,
                 n_classes=6,
                 total_teachers=10,
                 max_teachers_per_class=3,
                 demand_scale=30,   # average students per class
                 p_absence=0.05,
                 max_weeks=12,
                 seed: Optional[int] = None):
        
        super().__init__()
        
        self.n_classes = n_classes
        self.total_teachers = total_teachers
        self.max_teachers_per_class = max_teachers_per_class
        self.demand_scale = demand_scale
        self.p_absence = p_absence
        self.max_weeks = max_weeks

        self.all_patterns = self._generate_patterns()
        self.action_space = spaces.Discrete(len(self.all_patterns))

        # Observation Space setup
        low = np.concatenate([
            np.zeros(n_classes),               
            np.zeros(n_classes),               
            np.array([0.0])                    
        ]).astype(np.float32)
        
        # Calculate a safe upper bound for normalized students
        max_normalized_students = 1000.0 / max(1, self.demand_scale)
        high = np.concatenate([
            np.full(n_classes, max_normalized_students), 
            np.full(n_classes, 1.0),                 
            np.array([total_teachers])               
        ]).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize the modern random number generator
        self.np_random = np.random.default_rng(seed)
        
        # Internal state variables (initialized in reset)
        self.week = 0
        self.students = None
        self.required_periods = None
        self.prev_coverage = None
        self.available_teachers = None
        
    def _generate_patterns(self):
        """Generates all feasible allocation patterns (simplified)."""
        patterns = []
        max_teachers_per_class = self.max_teachers_per_class
        total_teachers = self.total_teachers
        n_classes = self.n_classes
        
        for allocation in product(range(max_teachers_per_class + 1), repeat=n_classes):
            if sum(allocation) <= total_teachers:
                patterns.append(np.array(allocation))
        
        if not patterns:
             patterns.append(np.zeros(n_classes, dtype=np.int32))
             
        return patterns
    
    def _get_obs(self):
        """Constructs the observation vector."""
        # 1. Normalized Student Counts
        obs_students = self.students / max(1, self.demand_scale)
        # 2. Previous Coverage Rate
        obs_coverage = self.prev_coverage
        # 3. Available Teachers (raw count)
        obs_available = np.array([self.available_teachers], dtype=np.float32)
        
        return np.concatenate([obs_students, obs_coverage, obs_available]).astype(np.float32)


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state, returning (observation, info).
        
        FIX 1: Updated signature to return (obs, info) for Gymnasium compatibility.
        """
        # Call super().reset() for handling seeding
        super().reset(seed=seed)
        
        self.week = 0
        self.students = self.np_random.poisson(self.demand_scale, size=self.n_classes)
        self.required_periods = np.full(self.n_classes, 10, dtype=np.int32)
        self.prev_coverage = np.zeros(self.n_classes, dtype=np.float32)
        self.available_teachers = self.total_teachers
        
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step in the environment.
        (Gymnasium-compliant signature)
        """
        alloc = self.all_patterns[action]
        teacher_capacity = 5.0 
        
        # Simulate teacher absences
        absent_teachers = self.np_random.binomial(self.total_teachers, self.p_absence)
        self.available_teachers = self.total_teachers - absent_teachers
        
        allocated_periods = alloc * teacher_capacity
        covered_periods = np.minimum(allocated_periods, self.required_periods)
        coverage_rate = covered_periods.sum() / max(1, self.required_periods.sum())

        # Compute metrics
        uncovered = self.required_periods.sum() - covered_periods.sum()
        class_workloads = covered_periods 
        self.prev_coverage = covered_periods / self.required_periods
        inequity = np.var(class_workloads) / (teacher_capacity**2 + 1e-6)

        # Reward design
        w_cov = 1.0
        w_uncovered = 0.5
        w_ineq = 0.2
        reward = w_cov * coverage_rate - w_uncovered * (uncovered / max(1, self.required_periods.sum())) - w_ineq * inequity

        # update students (small fluctuation)
        # FIX 2: Using the modern .integers() instead of the deprecated .randint()
        self.students = np.clip(self.students + self.np_random.integers(-2, 3, size=self.n_classes), 0, 1000)

        self.week += 1
        terminated = (self.week >= self.max_weeks)
        truncated = False 

        info = {
            "coverage_rate": coverage_rate, 
            "uncovered": int(uncovered), 
            "alloc": alloc.tolist(), 
            "absences": int(absent_teachers)
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Required by API, but not implemented for this environment."""
        pass

    def close(self):
        """Required by API."""
        pass