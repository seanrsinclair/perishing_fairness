import numpy as np
import matplotlib.pyplot as plt

EPS = 10e-3
DEBUG = False
RETURN_MEAN = True



def average_inversion_given_sample(resource_perish, order):
    count_inversions = 0

    B = len(resource_perish)

    for b in range(B):
        for b_prime in range(B):
            if b == b_prime:
                continue
            if (resource_perish[b] < resource_perish[b_prime]) and (order[b] > order[b_prime]):
                    if DEBUG: print(f'Found a swap: resource {b} and {b_prime} with orders {order[b], order[b_prime]} and perishing {resource_perish[b], resource_perish[b_prime]}')
                    count_inversions += 1
    return count_inversions / (B*(B-1))




def prob_confidence_interval(avg_mass, std_mass, n):
    """
        Helper code for calculating a high probability upper estimate on
        \sum_b \Ind{E} where avg_mass = \sum_b \Pr{E} and n is 1 / delta, high probability value

        Mostly done to ease tuning confidence interval choices jointly
    """

    # return mean + np.log(n) + np.sqrt(mean * np.log(n))
    if DEBUG: print(f"Avg Mass: {avg_mass}")
    if DEBUG: print(f"Confidence Interval (1) : {avg_mass + (1/(2*avg_mass))*(np.log(n) + np.sqrt(np.log(n)**2 + 8 * avg_mass + np.log(n)))}, Confidence Interval (2): {avg_mass + np.sqrt(n*np.log(n))}, Confidence Interval (3): {avg_mass + np.sqrt(avg_mass * np.log(n))}")
    min_value = np.min([avg_mass + np.sqrt(std_mass*np.log(n)), avg_mass + np.sqrt(3*avg_mass * np.log(n)), avg_mass + (1/(2*avg_mass))*(np.log(n) + np.sqrt(np.log(n)**2 + 8 * avg_mass + np.log(n)))])
    if DEBUG: print(f"Minimum Value: {min_value}")
    if RETURN_MEAN:
        return avg_mass 
    else:
        return min_value

def export_legend(legend, filename="LABEL_ONLY.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    plt.close('all')




def n_upper(demand_dist, n, num_iters = 1000):
    """
        Returns high probability estimate for N_{\geq t} for each t
    """
    future_demand_samples = [[np.sum(demand_dist(n)[t:]) for t in range(n)] for _ in range(1000)] # gets samples of the total demand N_{> t}
    avg_demand = np.mean(future_demand_samples, axis=0)
    n_upper = avg_demand + np.asarray([np.sqrt(2*np.std(future_demand_samples, axis=0)[t]*(n-t)) for t in range(n)])
    return n_upper




def check_offset_expiry(perish_dist, demand_dist, n, max_budget, num_iters = 1000):
    """
        Takes as input a perishing and demand distribution and outputs the probability estimate
        that offset expiry is satisfied
    """
    num_valid = 0 
    for _ in range(num_iters):
        sizes = demand_dist(n)
        resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)])
        check_optimality = [(max_budget / np.sum(sizes))*np.sum(sizes[:(t+1)]) 
                                - np.count_nonzero([resource_perish <= t]) for t in range(n)]
        if np.min(check_optimality) >= 0: # checks if B/N is feasible in hindsight
            num_valid += 1
    return (num_valid / num_iters)


def x_lower_line_search(perish_dist, demand_dist, n, max_budget, n_upper, order):
    """
        Takes as input a perishing and demand distribution and outputs an estimate of
        X_lower, the largest feasible X taking into account perishing    
    """

    valid = False # Indicator for whether a valid solution has been found

    x_lower = max_budget / n_upper[0] + EPS # starts off as checking Delta(X) for X = B / N_upper (as in, the feasible X_lower with no perishing)
    if DEBUG: print(f"Starting value: {x_lower - EPS}")
    while not valid: # loops down from x_lower until a valid solution is found
        x_lower = x_lower - EPS # takes off epsilon to search for lower values
        delta = estimate_delta(x_lower, perish_dist, demand_dist, n, max_budget, order) # estimates Delta(X)
        if x_lower <= (max_budget - delta) / (n_upper[0]): # Checks if X <= (B - Delta) / n_upper
            valid = True
    if DEBUG: print(f"Final value: {x_lower}")
    return x_lower


def taub(b, alloc, order, n_upper):
    """
    Computes the earliest timestep (taub) at which resource unit `b` is expected to be allocated,
    assuming a fixed allocation rate per customer and a priority ordering over resources.

    The function determines when enough cumulative allocation has occurred to reach the position
    of resource `b` in the allocation order. Allocation is assumed to proceed in order over time,
    with `n_upper[t] * alloc` units allocated at each timestep `t`.

    Args:
        b (int): Resource index.
        alloc (float): Fixed allocation per customer at each timestep.
        order (list or np.array): Priority order over resources; `order[b]` gives the position (starting from 0)
                                  of resource `b` in the allocation sequence.
        n_upper (np.array): Array of predicted upper bounds on the number of customers at each timestep.

    Returns:
        int: The first timestep (1-indexed) when resource `b` is expected to be allocated.
             Returns `inf` if the cumulative allocation never reaches the resource.
    """

    threshold = 1+order[b]
    for t in range(n_upper.shape[0]):
        allocated = n_upper[t] * alloc
        if allocated >= threshold:
            return t + 1  # since t starts from 0 but time starts from 1
    return float('inf')  # if condition is never met


def estimate_delta(x, perish_dist, demand_dist, n, max_budget, order, num_iters=500):
    """
    Estimates the expected number of resources that perish before allocation under a
    fixed allocation rate `x`, using Monte Carlo simulation.

    This estimate corresponds to the expected value of Δ(X), where Δ(X) is the number
    of resource units that expire before being allocated given the allocation rule.

    The simulation works as follows:
    - For each iteration, generate perishing times for all resources using `perish_dist`.
    - Estimate the time of allocation for each resource unit using the taub(·) function,
      which depends on the order `order`, the allocation rate `x`, and demand upper bounds.
    - Count how many resources perish before they are expected to be allocated.
    - Return a high-confidence estimate of the mean number of such prematurely perished resources.

    Args:
        x (float): Fixed per-customer allocation rate.
        perish_dist (callable): Function b, n ↦ T_b, returns perishing time for resource b.
        demand_dist (callable): Function n ↦ list of demands N_t for t in [0, n-1].
        n (int): Number of time periods (horizon).
        max_budget (int): Total number of resources available at the beginning.
        order (list or np.array): Allocation priority order; order[b] gives the position of resource b in the schedule.
        num_iters (int): Number of Monte Carlo iterations to run.

    Returns:
        float: High-confidence estimate of Δ(X), computed via a confidence interval over simulation averages.
    """
    
    # Estimate the expected demand size across time (used in DEBUG output)
    mean = np.mean([np.mean(demand_dist(n)) for _ in range(num_iters)])

    # Precompute upper bounds on customer demand at each time step
    demand_upper = n_upper(demand_dist, n)

    # List to store perished resource counts across simulation runs
    total_mass = []

    for _ in range(num_iters):
        # Sample perishing times for each resource unit
        resource_perish = np.asarray([perish_dist(b, n) for b in range(max_budget)])

        # Count how many resources perish before their expected allocation time
        perished = np.sum([
            # 1 if resource_perish[b] < np.minimum(n, taub(b, x, order, demand_upper)) else 0
            1 if resource_perish[b] < np.minimum(n, np.ceil((order[b]+1)/(mean*x))) else 0
            for b in range(max_budget)
        ])
        total_mass.append(perished)

        # Optional debug logging of perishing vs allocation time
        if DEBUG:
            for b in range(max_budget):
                expected_allocation = np.ceil((1 + order[b]) / (mean * x))
                print(f"Perishing time: {resource_perish[b]}, allocation time: {min(n, expected_allocation)}")

    # Compute sample mean and standard deviation of Δ(X)
    avg_mass = np.mean(total_mass)
    std_mass = np.std(total_mass)

    # Return a high-confidence estimate (e.g., upper bound) using normal approximation
    return prob_confidence_interval(avg_mass, std_mass, n)




def perish_future(t, current_index, resource_dict, x_lower, perish_dist, demand_dist, n, max_budget, order, num_iters=100):
    """
    Estimates the number of resources likely to perish before being allocated under
    a fixed threshold x_lower policy, using Monte Carlo simulations.

    The estimate is state-dependent and respects the priority order over resources.

    Args:
        t (int): Current timestep.
        current_index (int): Pointer to next resource index in order to be considered for allocation.
        resource_dict (dict): Dictionary mapping resource ID to (fraction_allocated, perish_time).
        x_lower (float): Fixed per-customer allocation level.
        perish_dist (callable): Function b, n ↦ perish_time, used for sampling perishing times.
        demand_dist (callable): Function n ↦ list of N_t, used for sampling future demand sequences.
        n (int): Total time horizon.
        max_budget (int): Total number of resources.
        order (list or np.array): Resource allocation priority (e.g., np.argsort(sigma)).
        num_iters (int): Number of Monte Carlo samples to estimate expected perishing.

    Returns:
        Upper bound estimate (e.g., high-confidence quantile) on the number of
        available-but-doomed resources that will perish before being allocated.
    """

    total_mass = 0

    available_resources = sorted(
        [b for b in range(max_budget) if resource_dict[str(b)][1] >= t and resource_dict[str(b)][0] < 1],
        key=lambda b: order[b]
    )


    # Estimate expected demand size for future steps
    mean_demand = np.mean([np.mean(demand_dist(n)[t:]) for _ in range(num_iters)])

    total_mass_samples = []


    for _ in range(num_iters):
        # sizes = demand_dist(n) # samples demands N_t

        resource_perish = np.asarray([perish_dist(b,n) for b in range(max_budget)]) # samples a vector of perishing times

        total_mass_samples.append(np.sum([1 if (resource_perish[b] < np.minimum(n, t + np.ceil((1+available_resources.index(b))/(mean_demand*x_lower)))) and (resource_perish[b] >= t) else 0 for b in available_resources]))
            # loop over remaining resources and checks whether resource will perish before earliest it gets allocated
            # note that right hand side taub is done in state dependent way, since x_lower will be allocated starting at the current_index
            # (+1) is done again due to indexing issues in python


    avg_mass = np.mean(total_mass)
    std_mass = np.std(total_mass)
    return prob_confidence_interval(avg_mass, std_mass, n)
