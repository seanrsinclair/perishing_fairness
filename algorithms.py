import numpy as np
import helper

DEBUG = False

def fixed_threshold(size, resource_perish, max_budget, xopt, x_lower, n, order):
    """
    Fixed threshold allocation algorithm. Takes as input a fixed allocation value x_lower
    to allocate across all timesteps until running out of budget.

    Args:
        size: list of N_t values
        resource_perish: list of perishing times for each resource unit
        max_budget: total number of resource units
        xopt: optimal allocation value
        x_lower: fixed threshold allocation per user
        n: number of timesteps
        order: array-like of length max_budget, where order[b] is the allocation order, i.e. order[0] is the first resource allocated

    Returns:
        perish_un_allocate: sum of unallocated-perished items
        waste: B - sum_t N_t * x_t
        counterfactual_envy: |xopt - x_t|
        hindsight_envy: max_t |x_t - x_t'|
        stockout: number of timesteps where allocation fell short due to exhausted budget
    """


    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    stockout = 0  # Count of timesteps where allocation was insufficient


    for t in range(n):
        if flag == False:
            to_allocate = size[t] * x_lower
            alloc_tracker = 0
            
            for b_idx in np.arange(current_index, max_budget):
                b = order[b_idx]
                (frac, perish_time) = resource_dict[str(b)]
                current_index = b
                if perish_time >= t: # perishing in the future from this round
                    alloc_amt = min(1 - frac, to_allocate - alloc_tracker)
                    alloc_tracker += alloc_amt
                    resource_dict[str(b)] = (frac+alloc_amt, perish_time)
                if alloc_tracker >= to_allocate:
                    break

            if alloc_tracker < to_allocate: # run out of resources
                if DEBUG: print(f'FIXED THRESHOLD: Out of resources at timestep: {t}')
                stockout = 1
                flag = True
                waste = 0
                if t != n-1:
                    counterfactual_envy = xopt
                    hindsight_envy = x_lower
                else:    
                    counterfactual_envy = np.abs(xopt - alloc_tracker / size[t])
                    hindsight_envy = np.abs(x_lower - alloc_tracker / size[t])

    if flag == False: # did not run out of resources
        hindsight_envy = 0
        counterfactual_envy = np.abs(xopt - x_lower)
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_allocate = np.sum([0 if resource_dict[str(b)][1] < n else (1 - resource_dict[str(b)][0]) for b in range(max_budget)])

    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout


def hope_guardrail_perish(size, resource_perish, max_budget, xopt, x_lower, n, Lt, demand_dist, perish_dist, n_upper, order):
    """
    Implements the perishing-aware guardrail allocation algorithm.

    This algorithm oscillates between a conservative allocation level `x_lower` and a more aggressive
    level `x_upper = x_lower + Lt`, depending on feasibility. Unlike the original guardrail policy,
    it includes a correction for resources expected to perish in the future when deciding whether
    allocating at the higher level is sustainable.

    Key characteristics:
    1. At each timestep t, the algorithm checks whether the remaining budget minus future demand
       (including a correction for perishability) supports allocating x_upper now and x_lower
       thereafter. If not, it allocates x_lower.
    2. Allocation proceeds greedily according to a resource order `order`, interpreted as increasing
       priority (e.g., increasing sigma).
    3. Perished resources (those with perish time < t) are skipped during allocation.
    4. The allocation halts when not enough viable resources remain to satisfy the desired level.

    Args:
        size (list): List of customer group sizes N_t at each timestep t.
        resource_perish (list): List of perishing times for each resource unit.
        max_budget (int): Total number of resource units available at the start.
        xopt (float): Offline optimal per-customer allocation benchmark for envy comparison.
        x_lower (float): Lower threshold for per-customer allocation.
        n (int): Number of timesteps (planning horizon).
        Lt (float): Guardrail lift; defines upper threshold x_upper = x_lower + Lt.
        demand_dist: Demand distribution parameters (passed to perish correction function).
        perish_dist: Perishability distribution parameters (passed to perish correction function).
        n_upper (list): Upper bounds on demand in future periods, used for lookahead feasibility.
        order (list or np.array): Allocation priority over resource units; a permutation of [0, ..., B-1].

    Returns:
        perish_un_allocate (float): Total unallocated fraction of units that did not perish by end of horizon.
        waste (float): Remaining unused budget at the end of the horizon.
        counterfactual_envy (float): |xopt - x_t| at the last feasible timestep.
        hindsight_envy (float): Max difference between any two realized allocations x_t and x_t'.
    """

    stockout = 0

    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    
    x_upper = x_lower + Lt
    
    if DEBUG: print(f"X_opt: {xopt}, X_lower: {x_lower}, X_upper: {x_upper}")
    
    current_budget = max_budget

    for t in range(n):
        if flag == False: # only continue when the algorithm has not run out of resources

            if t != n-1:
                n_upper_future = n_upper[t+1]
                perish_future = helper.perish_future(t, current_index, resource_dict, x_lower, perish_dist, demand_dist, n, max_budget, order)        
            else:
                perish_future = 0
                n_upper_future = 0


            if DEBUG: print(f"t: {t}, current_budget: {current_budget}, n_upper: {n_upper_future}, perish_future {perish_future}")
            if DEBUG: print(f"Check without perish: {size[t]*x_upper + n_upper_future*x_lower}")

            if current_budget - size[t]*x_upper - n_upper_future*x_lower - perish_future >= 0:
                to_allocate = size[t] * x_upper
                if DEBUG: print(f"t: {t} allocating x_upper")
            else:
                to_allocate = size[t] * x_lower
            
            alloc_tracker = 0
            
            for b_idx in np.arange(current_index, max_budget):
                b = order[b_idx]
                (frac, perish_time) = resource_dict[str(b)]
                current_index = b
                if perish_time >= t: # perishing in the future from this round
                    alloc_amt = min(1 - frac, to_allocate - alloc_tracker)
                    alloc_tracker += alloc_amt
                    resource_dict[str(b)] = (frac+alloc_amt, perish_time)
                if alloc_tracker >= to_allocate:
                    break

            # Update current budget
            current_budget = max_budget - np.sum([1 if resource_dict[str(b)][1] <= t else resource_dict[str(b)][0] for b in range(max_budget)])
            if alloc_tracker < to_allocate: # run out of resources
                if DEBUG: print(f'PERISH GUARDRAIL: Out of resources at timestep: {t} versus horizon {n}')
                flag = True
                waste = 0
                stockout = 1
                if t != n-1:
                    counterfactual_envy = xopt
                    hindsight_envy = x_upper
                else:    
                    counterfactual_envy = np.abs(xopt - alloc_tracker / size[t])
                    hindsight_envy = np.abs(x_upper - alloc_tracker / size[t])

    if flag == False: # did not run out of resources
        hindsight_envy = np.abs(x_upper - x_lower)
        counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper)])
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_allocate = np.sum([0 if resource_dict[str(b)][1] < n else (1 - resource_dict[str(b)][0]) for b in range(max_budget)])
    
    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout



def hope_guardrail_original(size, resource_perish, max_budget, xopt, x_lower, n, Lt, demand_dist, perish_dist, n_upper, order):
    """
    Implements the original guardrail allocation algorithm with a fixed lower threshold (x_lower)
    and an upper threshold (x_upper = x_lower + L_t), based on a lookahead feasibility check.

    This policy oscillates between x_lower and x_upper at each time step, depending on whether the
    remaining resource budget is sufficient to allocate x_upper now and x_lower for all future periods.

    Key characteristics:
    1. x_lower is set deterministically as max_budget / n_upper[0], and does not account for perishing.
    2. The decision to allocate x_upper at time t is made without considering perishing in future periods.
    3. Resources are allocated in the order specified by `order`, interpreted as increasing priority (e.g., increasing sigma).
    4. Resources that perish before time t are excluded from allocation at time t.

    Args:
        size (list): List of demand sizes N_t over the horizon.
        resource_perish (list): List of perishing times for each resource unit.
        max_budget (int): Total initial number of available resource units.
        xopt (float): Offline optimal per-unit allocation benchmark for envy comparison.
        x_lower (float): Placeholder (overwritten) for lower threshold allocation.
        n (int): Time horizon (number of timesteps).
        Lt (float): Guardrail lift that defines x_upper = x_lower + Lt.
        demand_dist, perish_dist: Unused arguments (legacy).
        n_upper (list): Future upper-bound estimates on demand, used for feasibility checks.
        order (list or np.array): Allocation priority over resource units; should be a permutation of [0, ..., B-1].

    Returns:
        perish_un_allocate (float): Total unallocated fraction of items that did not perish by horizon end.
        waste (float): Total unused budget at the end of the horizon.
        counterfactual_envy (float): Worst-case difference between xopt and realized allocation across all t.
        hindsight_envy (float): Max difference between any two realized allocations x_t and x_t'.
    """

    resource_dict = {}
    for b in range(max_budget): # Initializes dictionary of the various resources and allocation
        resource_dict[str(b)] = (0, resource_perish[b])
    flag = False  # Flag for running out of budget
    current_index = 0
    stockout = 0
    x_lower = max_budget / n_upper[0]

    x_upper = x_lower + Lt
    if DEBUG: print(f"X_lower: {x_lower}, X_upper: {x_upper}")
    
    current_budget = max_budget

    for t in range(n):
        if flag == False:
                
            if t != n-1:
                n_upper_future = n_upper[t+1]
            else:
                n_upper_future = 0


            if DEBUG: print(f"t: {t}, current_budget: {current_budget}, n_upper: {n_upper_future}")
            if DEBUG: print(f"Check without perish: {size[t]*x_upper + n_upper_future*x_lower}")

            if current_budget - size[t]*x_upper - n_upper_future*x_lower >= 0:
                to_allocate = size[t] * x_upper
                if DEBUG: print(f"t: {t} allocating x_upper")
            else:
                to_allocate = size[t] * x_lower
            
            alloc_tracker = 0
            
            for b_idx in np.arange(current_index, max_budget):
                b = order[b_idx]
                (frac, perish_time) = resource_dict[str(b)]
                current_index = b
                if perish_time >= t: # perishing in the future from this round
                    alloc_amt = min(1 - frac, to_allocate - alloc_tracker)
                    alloc_tracker += alloc_amt
                    resource_dict[str(b)] = (frac+alloc_amt, perish_time)
                if alloc_tracker >= to_allocate:
                    break


            # Update current budget
            current_budget = max_budget - np.sum([1 if resource_dict[str(b)][1] <= t else resource_dict[str(b)][0] for b in range(max_budget)])
            if alloc_tracker < to_allocate: # run out of resources
                if DEBUG: print(f'OG GUARDRAIL: Out of resources at timestep: {t} versus horizon {n}')
                flag = True
                waste = 0
                stockout = 1
                if t != n-1:
                    counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper), np.abs(xopt)])
                    hindsight_envy = x_upper

                else:    
                    counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper), np.abs(xopt), np.max(xopt - (alloc_tracker / size[t]))])
                    hindsight_envy = np.max([np.abs(x_upper - (alloc_tracker / size[t])), np.abs(x_upper - x_lower)])
                    if DEBUG: print(f"To Allocate: {to_allocate}, Allocated: {alloc_tracker}")
                

    if flag == False: # did not run out of resources
        hindsight_envy = np.abs(x_upper - x_lower)
        counterfactual_envy = np.max([np.abs(xopt - x_lower), np.abs(xopt - x_upper)])
        waste = max_budget - np.sum([resource_dict[str(b)][0] for b in range(max_budget)])

    perish_un_allocate = np.sum([0 if resource_dict[str(b)][1] < n else (1 - resource_dict[str(b)][0]) for b in range(max_budget)])
    
    return perish_un_allocate, waste, counterfactual_envy, hindsight_envy, stockout