import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Union
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
        
        # Initialize the base class
        super().__init__()
        
        self.n_classes = n_classes
        self.total_teachers = total_teachers
        self.max_teachers_per_class = max_teachers_per_class
        self.demand_scale = demand_scale
        self.p_absence = p_absence
        self.max_weeks = max_weeks

        # Precompute a set of feasible allocation patterns to keep action space small.
        self.all_patterns = self._generate_patterns()
        self.action_space = spaces.Discrete(len(self.all_patterns))

        # --- FIX IMPLEMENTATION (Option 1): Adjusting Observation Bounds ---
        # The students are generated in reset() with max value of int(demand_scale * 1.2) + 1
        max_student_demand_generated = int(self.demand_scale * 1.2) + 1
        max_normalized_demand = max_student_demand_generated / self.demand_scale

        # Observation space definition:
        # [0:n_classes-1]: Normalized student count (demand) - uses max_normalized_demand
        # [n_classes:2*n_classes-1]: Previous week's coverage (state memory) - max 1.0
        # [2*n_classes]: Available teachers (resource level) - max total_teachers/max_teachers_per_class
        low = np.zeros(2 * self.n_classes + 1, dtype=np.float32)
        high = np.array(
            [max_normalized_demand] * self.n_classes +  # Fixed boundary here
            [1.0] * self.n_classes + 
            [self.total_teachers / self.max_teachers_per_class], 
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Internal state variables (initialized in reset)
        self.students = None
        self.week = 0
        self.required_periods = None
        self.prev_coverage = None
        
    def _generate_patterns(self):
        """Generates all feasible allocation patterns (discrete actions)."""
        
        # Generate all combinations up to max_teachers_per_class per class
        all_possible_allocations = list(product(range(self.max_teachers_per_class + 1), repeat=self.n_classes))
        
        # Filter for patterns where the total allocation is within budget
        feasible_patterns = [
            list(pattern) 
            for pattern in all_possible_allocations 
            if sum(pattern) <= self.total_teachers
        ]
        
        return np.array(feasible_patterns, dtype=np.int32)
    
    def _get_obs(self):
        """Computes the observation vector."""
        # Note: Normalization remains the same, but the observation space now supports values > 1.0
        obs_students_normalized = self.students / self.demand_scale
        obs_available_teachers_normalized = self.total_teachers / self.max_teachers_per_class
        
        # Ensure prev_coverage is within [0, 1]
        prev_coverage_clipped = np.clip(self.prev_coverage, 0.0, 1.0)
        
        return np.concatenate([
            obs_students_normalized,
            prev_coverage_clipped,
            np.array([obs_available_teachers_normalized])
        ]).astype(np.float32)
        
    def _get_info(self):
        """Returns extra diagnostic information (for Gymnasium API)."""
        return {
            "week": self.week,
            "students": self.students.tolist(),
            "required_periods": self.required_periods.tolist()
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """Resets the environment state."""
        # 1. Call super().reset() for seeding
        super().reset(seed=seed)
        
        # 2. Reset internal state
        self.week = 0
        
        # Initial student count (demand)
        self.students = self.np_random.integers(
            low=int(self.demand_scale * 0.8),
            high=int(self.demand_scale * 1.2) + 1, 
            size=self.n_classes,
            dtype=np.int32
        )
        
        # Initial required periods (proxy for required teaching load based on demand)
        # Assuming 1 required period per 10 students, max 10 periods
        self.required_periods = np.clip(
            (self.students / 10).astype(np.int32), 
            a_min=1, 
            a_max=10
        )
        
        # Initial coverage is assumed to be perfect (1.0)
        self.prev_coverage = np.ones(self.n_classes, dtype=np.float32)
        
        return self._get_obs(), self._get_info()

    def step(self, action: Union[int, np.ndarray]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Performs one step in the environment."""
        
        # --- Decode Action ---
        if isinstance(action, np.ndarray):
            action = int(action.item())
        alloc = self.all_patterns[action]
        
        # --- Simulate Absenteeism and Teacher Capacity ---
        # Teacher capacity (e.g., periods per teacher per week)
        teacher_capacity = 5 
        
        # Simulate teacher absences
        absences = self.np_random.binomial(n=self.total_teachers, p=self.p_absence)
        actual_teachers = self.total_teachers - absences
        
        # Use a copy of the allocation for modification during shortage simulation
        current_alloc = alloc.copy()
        required_total = current_alloc.sum()
        
        # If the allocated teachers exceed the available pool (due to absences or over-allocation)
        if required_total > actual_teachers:
            # We must reduce the allocation until total_teachers <= actual_teachers.
            diff = required_total - actual_teachers
            
            # Reduce from the class with the highest current allocation until budget is met.
            # We need a loop to handle distributed reduction.
            while diff > 0 and current_alloc.sum() > actual_teachers:
                # Find the class with the highest current allocation (use min to handle ties consistently)
                # The reduction logic needs to handle the fact that we can't reduce below 0.
                highest_idx = np.argmax(current_alloc)
                
                # Reduction amount is 1, but we can't reduce below 0.
                if current_alloc[highest_idx] > 0:
                    current_alloc[highest_idx] -= 1
                    diff -= 1
                else:
                    # If the highest allocated class is 0, we can stop reducing this way.
                    # This only happens if all allocations are 0, which means required_total was 0, 
                    # but the IF condition required_total > actual_teachers implies required_total > 0.
                    # We break if we can't reduce any further (e.g. all are 0)
                    if np.all(current_alloc == 0):
                        break
                    # If we hit 0, we search for the next highest, but argmax handles this.
                    # This block is mostly a safety net.
                    break 

        alloc_after_shortage = current_alloc
        
        # --- Compute Coverage ---
        # The maximum number of periods covered by the allocation (alloc * capacity)
        max_covered_periods = alloc_after_shortage * teacher_capacity 
        
        # Actual covered periods is limited by the actual required periods
        covered_periods = np.minimum(max_covered_periods, self.required_periods)
        
        total_required = self.required_periods.sum()
        
        # Handle division by zero if all demand is zero (shouldn't happen with a_min=1, but for safety)
        if total_required == 0:
            coverage_rate = 1.0
            uncovered = 0
            uncovered_ratio = 0.0
        else:
            coverage_rate = covered_periods.sum() / total_required
            uncovered = total_required - covered_periods.sum()
            uncovered_ratio = uncovered / total_required
        
        # Workload (proxy: actual periods covered)
        class_workloads = covered_periods  
        self.prev_coverage = covered_periods / self.required_periods  # per-class ratio for next state
        
        # Inequity penalty: variance of class workloads (normalized by max possible squared workload)
        inequity = np.var(class_workloads) / (self.max_teachers_per_class * teacher_capacity)**2

        # Reward design
        w_cov = 1.0     # Weight for overall coverage
        w_uncovered = 0.5 # Weight for uncovered demand
        w_ineq = 0.2    # Weight for equity penalty
        
        reward = w_cov * coverage_rate - w_uncovered * uncovered_ratio - w_ineq * inequity

        # --- Update State ---
        # Update students (small random fluctuation)
        self.students = np.clip(
            self.students + self.np_random.integers(-2, 3, size=self.n_classes), 
            a_min=1, # Ensure minimum student count is 1 to avoid zero required periods
            a_max=1000
        )
        
        # Recompute required periods based on new student count
        self.required_periods = np.clip(
            (self.students / 10).astype(np.int32), 
            a_min=1, 
            a_max=10
        )

        self.week += 1
        
        # Check termination and truncation
        terminated = (self.week >= self.max_weeks)
        truncated = False 
        
        info = {
            "coverage_rate": coverage_rate, 
            "uncovered": int(uncovered), 
            "alloc": alloc_after_shortage.tolist(), 
            "absences": int(absences),
            "terminated": terminated,
            "truncated": truncated
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Required by API, but not implemented for this environment."""
        pass

    def close(self):
        """Required by API."""
        pass


# --- Training Script ---
if __name__ == "__main__":
    print("[INFO] Training started.")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("[ERROR] stable_baselines3 is not installed. Please install it with 'pip install stable-baselines3'.")
        exit(1)

    # Create environment
    env = TeacherAllocEnv()
    
    # Check the environment: This should now pass without assertion errors
    # because the observation space bounds correctly reflect the possible 
    # normalized student count (up to 1.233...).
    print("[INFO] Checking environment consistency...")
    check_env(env, warn=True)
    print("[INFO] Environment check passed.")


    # Create PPO model
    # Note: Using MlpPolicy for the discrete action space
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    # Reduced timesteps for quicker testing/demonstration
    model.learn(total_timesteps=100000)

    # Save the model
    import os
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/ppo_teacher_alloc.zip")
    print("[INFO] Training finished. Model saved to 'saved_models/ppo_teacher_alloc.zip'.")