"""Drone Search Policy — 102483055.py"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np


@dataclass
class SurvivorCandidate:
    id: int
    x: float
    y: float
    count: int = 1
    triage: Optional[str] = None
    triage_counts: Dict[str, int] = field(default_factory=lambda: {"critical": 0, "serious": 0, "minor": 0})
    seen_steps: Set[int] = field(default_factory=set)
    is_declared: bool = False
    needs_investigation: bool = True


class SearchMemory:
    """Tracks arena prior probabilities and real-time sensor coverage."""
    def __init__(self, arena_size: float = 400.0, grid_size: int = 40, cell_size: float = 10.0,
                 prior_map: Optional[List[List[float]]] = None):
        self.arena_size = arena_size
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        self.xs = (np.arange(grid_size) + 0.5) * cell_size
        self.ys = (np.arange(grid_size) + 0.5) * cell_size
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys, indexing="ij")
        
        if prior_map is not None:
            self.prior = np.array(prior_map, dtype=np.float64)
            prior_sum = self.prior.sum()
            self.prior_norm = self.prior / prior_sum if prior_sum > 0 else np.full((grid_size, grid_size), 1.0 / (grid_size * grid_size))
        else:
            self.prior = np.ones((grid_size, grid_size), dtype=np.float64)
            self.prior_norm = np.full((grid_size, grid_size), 1.0 / (grid_size * grid_size))

        self.coverage = np.zeros((grid_size, grid_size), dtype=np.float64)

    def mark_sweep(self, x0: float, y0: float, x1: float, y1: float, altitude: float, footprint_tan: float):
        """Updates coverage along flight leg using localized bounding boxes."""
        sensor_radius = max(5.0, altitude * footprint_tan)
        dist = math.hypot(x1 - x0, y1 - y0)
        num_samples = max(2, int(math.ceil(dist / (self.cell_size * 0.75))))
        alphas = np.linspace(0.0, 1.0, num_samples)
        sample_xs = x0 + alphas * (x1 - x0)
        sample_ys = y0 + alphas * (y1 - y0)
        r_sq = sensor_radius * sensor_radius

        for sx, sy in zip(sample_xs, sample_ys):
            min_i = max(0, int((sx - sensor_radius) / self.cell_size))
            max_i = min(self.grid_size - 1, int((sx + sensor_radius) / self.cell_size))
            min_j = max(0, int((sy - sensor_radius) / self.cell_size))
            max_j = min(self.grid_size - 1, int((sy + sensor_radius) / self.cell_size))
            
            sub_gx = self.grid_x[min_i:max_i+1, min_j:max_j+1]
            sub_gy = self.grid_y[min_i:max_i+1, min_j:max_j+1]
            mask = ((sub_gx - sx)**2 + (sub_gy - sy)**2) <= r_sq
            
            sub_cov = self.coverage[min_i:max_i+1, min_j:max_j+1]
            sub_cov[mask] = np.minimum(1.0, sub_cov[mask] + 0.45)

    def get_unsearched_priority_map(self) -> np.ndarray:
        """Returns array of unsearched utility; covered cells drop to 0 to prevent loops."""
        unsearched_weight = np.where(self.coverage >= 0.45, 0.0, 1.0 - self.coverage)
        return self.prior_norm * unsearched_weight


class SurvivorTracker:
    """Clusters noisy detections, tracks observation counts, and deduplicates declarations."""
    def __init__(self, merge_radius: float = 9.0, declaration_dedup_radius: float = 12.0):
        self.merge_radius = merge_radius
        self.declaration_dedup_radius = declaration_dedup_radius
        self.candidates: List[SurvivorCandidate] = []
        self._next_id = 1
        self.declared_locations: List[Tuple[float, float, str]] = []

    def update(self, detections: List[Dict[str, Any]], current_step: int):
        """Processes incoming detections into clustered candidate entities."""
        for det in detections:
            dx = float(det["x"])
            dy = float(det["y"])
            dtriage = det.get("triage")

            if self.is_near_declared(dx, dy, self.declaration_dedup_radius):
                continue

            best_cand: Optional[SurvivorCandidate] = None
            min_dist = float("inf")
            for cand in self.candidates:
                dist = math.hypot(cand.x - dx, cand.y - dy)
                if dist <= self.merge_radius and dist < min_dist:
                    min_dist = dist
                    best_cand = cand

            if best_cand is not None:
                weight = 1.0 / (best_cand.count + 1)
                best_cand.x = (1.0 - weight) * best_cand.x + weight * dx
                best_cand.y = (1.0 - weight) * best_cand.y + weight * dy
                best_cand.count += 1
                best_cand.seen_steps.add(current_step)

                if dtriage in best_cand.triage_counts:
                    best_cand.triage_counts[dtriage] += 1
                    best_cand.triage = max(best_cand.triage_counts, key=best_cand.triage_counts.get)
                    best_cand.needs_investigation = False
            else:
                cand = SurvivorCandidate(
                    id=self._next_id,
                    x=dx,
                    y=dy,
                    count=1,
                    triage=dtriage,
                    is_declared=False,
                    needs_investigation=(dtriage is None)
                )
                cand.seen_steps.add(current_step)
                if dtriage in cand.triage_counts:
                    cand.triage_counts[dtriage] += 1
                self.candidates.append(cand)
                self._next_id += 1

    def is_near_declared(self, x: float, y: float, radius: float) -> bool:
        """Returns True if (x, y) is within radius of any already declared survivor."""
        r_sq = radius * radius
        for dx, dy, _ in self.declared_locations:
            if (x - dx)**2 + (y - dy)**2 <= r_sq:
                return True
        return False

    def get_ready_to_declare(self) -> List[SurvivorCandidate]:
        """Returns candidates verified by triage, multiple actions, or high detection count."""
        ready = []
        for cand in self.candidates:
            if cand.is_declared:
                continue
            if self.is_near_declared(cand.x, cand.y, self.declaration_dedup_radius):
                cand.is_declared = True
                continue
            if cand.triage is not None or len(cand.seen_steps) >= 2 or cand.count >= 3:
                ready.append(cand)
        return ready

    def get_investigation_candidates(self) -> List[SurvivorCandidate]:
        """Returns unverified candidates needing low-altitude confirmation."""
        needed = []
        for cand in self.candidates:
            if not cand.is_declared and cand.needs_investigation:
                if not self.is_near_declared(cand.x, cand.y, self.declaration_dedup_radius):
                    needed.append(cand)
        return needed

    def mark_declared(self, cand_id: int, x: float, y: float, triage: str):
        """Marks candidate declared and suppresses nearby duplicates within dedup perimeter."""
        for cand in self.candidates:
            if cand.id == cand_id or math.hypot(cand.x - x, cand.y - y) <= self.declaration_dedup_radius:
                cand.is_declared = True
                cand.needs_investigation = False
        self.declared_locations.append((x, y, triage))


class TriageManager:
    """Evaluates time-decay curves and assigns optimal declaration triage values."""
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 decay_tau: Optional[Dict[str, float]] = None,
                 wrong_factor: float = 0.5):
        self.weights = weights or {"critical": 3.0, "serious": 2.0, "minor": 1.0}
        self.decay_tau = decay_tau or {"critical": 300.0, "serious": 600.0, "minor": 1200.0}
        self.wrong_factor = wrong_factor
        self.default_fallback = "serious"

    def estimate_potential_value(self, triage: Optional[str], current_t: float) -> float:
        """Computes expected time-discounted score value for a triage category."""
        target_triage = triage if (triage in self.weights) else self.default_fallback
        w = self.weights[target_triage]
        tau = self.decay_tau[target_triage]
        mult = 1.0 if triage is not None else self.wrong_factor
        return w * math.exp(-current_t / tau) * mult

    def resolve_declaration_triage(self, cand: SurvivorCandidate) -> str:
        """Resolves best triage string for declaration."""
        if cand.triage in self.weights:
            return cand.triage
        if any(cand.triage_counts.values()):
            best = max(cand.triage_counts, key=cand.triage_counts.get)
            if cand.triage_counts[best] > 0:
                return best
        return self.default_fallback


class BatteryManager:
    """Monitors power consumption and determines safe return threshold."""
    def __init__(self, total_budget: float, base_xy: Tuple[float, float] = (0.0, 0.0),
                 safety_factor: float = 1.45, fixed_margin_pct: float = 0.05):
        self.total_budget = total_budget
        self.base_x, self.base_y = base_xy
        self.safety_factor = safety_factor
        self.fixed_reserve = total_budget * fixed_margin_pct
        self.power_ema: float = 3.0
        self.alpha: float = 0.25

    def update_energy(self, energy_used: float, dt: float, action_type: str = "fly_to"):
        """Updates moving average of flight power draw."""
        if dt > 0.05 and energy_used > 0.0 and action_type == "fly_to":
            current_power = energy_used / dt
            self.power_ema = (1.0 - self.alpha) * self.power_ema + self.alpha * current_power

    def estimate_return_cost(self, x: float, y: float, cruise_speed: float = 10.5) -> float:
        """Estimates worst-case return flight energy including safety margin."""
        dist_home = math.hypot(x - self.base_x, y - self.base_y)
        t_fly = dist_home / cruise_speed
        effective_power = max(self.power_ema, 2.5)
        return (t_fly * effective_power * self.safety_factor) + self.fixed_reserve

    def must_return_now(self, x: float, y: float, energy_remaining: float, cruise_speed: float = 10.5) -> bool:
        """Returns True if battery is required for safe landing."""
        return energy_remaining <= self.estimate_return_cost(x, y, cruise_speed)


class FlightController:
    """Validates parameters and constructs bounded action dictionaries."""
    def __init__(self, arena_size: float = 400.0, altitude_range: Tuple[float, float] = (5.0, 80.0),
                 speed_range: Tuple[float, float] = (1.0, 15.0), land_radius: float = 15.0):
        self.arena_size = arena_size
        self.min_alt, self.max_alt = altitude_range
        self.min_spd, self.max_spd = speed_range
        self.land_radius = land_radius

    def fly_to(self, x: float, y: float, altitude: float, speed: float) -> Dict[str, Any]:
        return {
            "type": "fly_to",
            "x": float(np.clip(x, 0.0, self.arena_size)),
            "y": float(np.clip(y, 0.0, self.arena_size)),
            "altitude": float(np.clip(altitude, self.min_alt, self.max_alt)),
            "speed": float(np.clip(speed, self.min_spd, self.max_spd))
        }

    def hover(self, duration: float) -> Dict[str, Any]:
        return {"type": "hover", "duration": float(np.clip(duration, 0.5, 30.0))}

    def declare(self, x: float, y: float, triage: str) -> Dict[str, Any]:
        valid_triage = triage if triage in ("critical", "serious", "minor") else "serious"
        return {
            "type": "declare",
            "x": float(np.clip(x, 0.0, self.arena_size)),
            "y": float(np.clip(y, 0.0, self.arena_size)),
            "triage": valid_triage
        }

    def land(self) -> Dict[str, Any]:
        return {"type": "land"}

    def can_land(self, x: float, y: float) -> bool:
        return math.hypot(x, y) <= (self.land_radius - 1.0)


class SearchPlanner:
    """Selects exploration targets based on unsearched prior density."""
    def __init__(self, memory: SearchMemory, arena_size: float = 400.0,
                 search_altitude: float = 40.0, search_speed: float = 10.5,
                 footprint_tan: float = 0.839):
        self.memory = memory
        self.arena_size = arena_size
        self.search_altitude = search_altitude
        self.search_speed = search_speed
        self.sensor_radius = search_altitude * footprint_tan
        self.sector_data = self._init_macro_sectors()

    def _init_macro_sectors(self) -> List[Tuple[float, float, np.ndarray]]:
        """Precomputes sector coordinates and spatial masks."""
        step = max(35.0, self.sensor_radius * 1.35)
        coords = np.arange(step * 0.5, self.arena_size, step)
        sector_data = []
        r_sq = (self.sensor_radius * 1.5)**2
        
        for i, x in enumerate(coords):
            ys = coords if (i % 2 == 0) else coords[::-1]
            for y in ys:
                fx, fy = float(x), float(y)
                mask = ((self.memory.grid_x - fx)**2 + (self.memory.grid_y - fy)**2) <= r_sq
                sector_data.append((fx, fy, mask))
        return sector_data

    def select_next_waypoint(self, current_x: float, current_y: float) -> Tuple[float, float, float, float]:
        """Dynamically chooses best waypoint balancing unsearched utility against distance."""
        unsearched_map = self.memory.get_unsearched_priority_map()
        best_score = -1.0
        best_target = (self.arena_size * 0.5, self.arena_size * 0.5)

        for sx, sy, mask in self.sector_data:
            utility = float(unsearched_map[mask].sum())
            if utility <= 1e-6:
                continue

            dist = math.hypot(sx - current_x, sy - current_y)
            score = utility / (1.0 + 0.0006 * dist)
            if score > best_score:
                best_score = score
                best_target = (sx, sy)

        if best_score <= 1e-6:
            max_idx = np.unravel_index(np.argmax(unsearched_map), unsearched_map.shape)
            best_target = (float(self.memory.xs[max_idx[0]]), float(self.memory.ys[max_idx[1]]))

        return best_target[0], best_target[1], self.search_altitude, self.search_speed


class Policy:
    """Mission controller orchestrating subsystems via a finite state machine."""
    def __init__(self, info: Dict[str, Any]):
        self.info = info
        self.arena_size = float(info.get("arena_size", 400.0))
        self.grid_size = int(info.get("grid_size", 40))
        self.cell_size = float(info.get("cell_size", 10.0))
        self.energy_budget = float(info.get("energy_budget", 1000.0))
        self.footprint_tan = float(info.get("footprint_tan", math.tan(math.radians(40.0))))
        self.land_radius = float(info.get("land_radius", 15.0))
        self.base_xy = (float(info.get("base_xy", [0.0, 0.0])[0]), float(info.get("base_xy", [0.0, 0.0])[1]))

        self.memory = SearchMemory(self.arena_size, self.grid_size, self.cell_size, info.get("prior_map"))
        self.tracker = SurvivorTracker(merge_radius=9.0, declaration_dedup_radius=float(info.get("match_radius", 5.0)) * 2.2)
        self.triage_mgr = TriageManager(info.get("triage_weights"), info.get("triage_decay_tau"), float(info.get("wrong_triage_factor", 0.5)))
        self.battery_mgr = BatteryManager(self.energy_budget, self.base_xy, safety_factor=1.45, fixed_margin_pct=0.05)
        self.controller = FlightController(self.arena_size, tuple(info.get("altitude_range", [5.0, 80.0])),
                                           tuple(info.get("speed_range", [1.0, 15.0])), self.land_radius)
        self.planner = SearchPlanner(self.memory, self.arena_size, search_altitude=40.0, search_speed=10.5,
                                     footprint_tan=self.footprint_tan)

        self.state = "INIT"
        self.steps = 0
        self.prev_x = self.base_xy[0]
        self.prev_y = self.base_xy[1]
        self.prev_t = 0.0
        self.last_action_type = "init"
        self.current_investigation_cand: Optional[SurvivorCandidate] = None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Main agent step: updates state models and returns exactly one action dict."""
        self.steps += 1
        curr_t = float(obs.get("t", 0.0))
        curr_x = float(obs.get("x", 0.0))
        curr_y = float(obs.get("y", 0.0))
        curr_alt = float(obs.get("altitude", 5.0))
        energy_remaining = float(obs.get("energy_remaining", 0.0))
        energy_used_last = float(obs.get("energy_used_last_action", 0.0))
        detections = obs.get("detections", [])

        dt = max(0.0, curr_t - self.prev_t)
        self.prev_t = curr_t

        self.battery_mgr.update_energy(energy_used_last, dt, self.last_action_type)

        if self.steps > 1 and self.last_action_type == "fly_to":
            self.memory.mark_sweep(self.prev_x, self.prev_y, curr_x, curr_y, curr_alt, self.footprint_tan)
        self.prev_x = curr_x
        self.prev_y = curr_y

        if detections:
            self.tracker.update(detections, self.steps)

        if self.battery_mgr.must_return_now(curr_x, curr_y, energy_remaining) and self.state not in ("RETURN", "LAND"):
            self.state = "RETURN"

        action = self._step_state_machine(curr_x, curr_y, curr_alt, curr_t)
        self.last_action_type = action.get("type", "fly_to")
        return action

    def _step_state_machine(self, curr_x: float, curr_y: float, curr_alt: float, curr_t: float) -> Dict[str, Any]:
        if self.state == "LAND":
            return self.controller.land()

        if self.state == "RETURN":
            if self.controller.can_land(curr_x, curr_y):
                self.state = "LAND"
                return self.controller.land()
            return self.controller.fly_to(self.base_xy[0], self.base_xy[1], altitude=25.0, speed=10.5)

        # 1. Immediate declaration of verified candidates
        ready_candidates = self.tracker.get_ready_to_declare()
        if ready_candidates:
            ready_candidates.sort(key=lambda c: self.triage_mgr.estimate_potential_value(c.triage, curr_t), reverse=True)
            target = ready_candidates[0]
            triage_str = self.triage_mgr.resolve_declaration_triage(target)
            self.tracker.mark_declared(target.id, target.x, target.y, triage_str)
            return self.controller.declare(target.x, target.y, triage_str)

        # 2. Close-range candidate investigation
        if self.state == "INVESTIGATE":
            if self.current_investigation_cand is not None and not self.current_investigation_cand.is_declared:
                cand = self.current_investigation_cand
                dist = math.hypot(curr_x - cand.x, curr_y - cand.y)
                if dist <= 12.0 and curr_alt <= 22.0:
                    triage_str = self.triage_mgr.resolve_declaration_triage(cand)
                    self.tracker.mark_declared(cand.id, cand.x, cand.y, triage_str)
                    self.current_investigation_cand = None
                    self.state = "SEARCH"
                    return self.controller.declare(cand.x, cand.y, triage_str)
                return self.controller.fly_to(cand.x, cand.y, altitude=18.0, speed=10.0)
            self.state = "SEARCH"
            self.current_investigation_cand = None

        inv_candidates = self.tracker.get_investigation_candidates()
        if inv_candidates:
            inv_candidates.sort(key=lambda c: math.hypot(curr_x - c.x, curr_y - c.y))
            nearest = inv_candidates[0]
            if math.hypot(curr_x - nearest.x, curr_y - nearest.y) <= 45.0:
                self.current_investigation_cand = nearest
                self.state = "INVESTIGATE"
                return self.controller.fly_to(nearest.x, nearest.y, altitude=18.0, speed=10.0)

        # 3. Macro search navigation
        self.state = "SEARCH"
        target_x, target_y, target_alt, target_spd = self.planner.select_next_waypoint(curr_x, curr_y)
        return self.controller.fly_to(target_x, target_y, target_alt, target_spd)
