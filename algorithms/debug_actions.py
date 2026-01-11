"""
Debug script to analyze why there are so many actions in the ILP.
"""
import sys
sys.path.insert(0, '/home/yimintan/research/WAFR2026')

from collections import defaultdict
import numpy as np

def analyze_trajectory(M):
    """Analyze a trajectory to understand action count."""
    N = len(M)
    T = len(M[0]) - 1

    print(f"=== Trajectory Analysis ===")
    print(f"N (agents): {N}")
    print(f"T (timesteps): {T}")
    print(f"Total positions: {N * (T + 1)}")

    # Count unique vertices per agent
    total_unique_vertices = 0
    total_actions = 0
    max_actions_agent = 0
    max_actions_agent_id = -1

    agent_stats = []

    for i in range(N):
        times_by_vertex = defaultdict(list)
        for k in range(T + 1):
            times_by_vertex[M[i][k]].append(k)

        num_unique_vertices = len(times_by_vertex)
        total_unique_vertices += num_unique_vertices

        # Count actions for this agent (same logic as enumerate_actions)
        agent_actions = 0
        for vertex, times in times_by_vertex.items():
            if len(times) < 2:
                continue

            # Compress consecutive times
            compressed_ts = []
            run_start = times[0]
            prev = times[0]
            for k in range(1, len(times)):
                if times[k] == prev + 1:
                    prev = times[k]
                else:
                    compressed_ts.append(run_start)
                    if prev != run_start:
                        compressed_ts.append(prev)
                    run_start = times[k]
                    prev = times[k]
            compressed_ts.append(run_start)
            if prev != run_start:
                compressed_ts.append(prev)

            L = len(compressed_ts)
            # Number of pairs = L * (L-1) / 2
            num_pairs = L * (L - 1) // 2
            agent_actions += num_pairs

        total_actions += agent_actions
        agent_stats.append({
            'agent': i,
            'unique_vertices': num_unique_vertices,
            'actions': agent_actions,
        })

        if agent_actions > max_actions_agent:
            max_actions_agent = agent_actions
            max_actions_agent_id = i

    print(f"\n=== Per-Agent Statistics ===")
    print(f"Total unique vertices across all agents: {total_unique_vertices}")
    print(f"Average unique vertices per agent: {total_unique_vertices / N:.1f}")
    print(f"Total actions: {total_actions}")
    print(f"Average actions per agent: {total_actions / N:.1f}")
    print(f"Max actions for single agent: {max_actions_agent} (agent {max_actions_agent_id})")

    # Show top 5 agents by action count
    agent_stats.sort(key=lambda x: x['actions'], reverse=True)
    print(f"\n=== Top 5 Agents by Action Count ===")
    for stat in agent_stats[:5]:
        print(f"  Agent {stat['agent']}: {stat['actions']} actions, {stat['unique_vertices']} unique vertices")

    # Analyze a specific agent with many actions
    if max_actions_agent_id >= 0:
        print(f"\n=== Detailed Analysis of Agent {max_actions_agent_id} ===")
        times_by_vertex = defaultdict(list)
        for k in range(T + 1):
            times_by_vertex[M[max_actions_agent_id][k]].append(k)

        print(f"Visits {len(times_by_vertex)} unique vertices:")
        vertex_stats = []
        for vertex, times in times_by_vertex.items():
            # Compress
            compressed_ts = []
            if len(times) >= 2:
                run_start = times[0]
                prev = times[0]
                for k in range(1, len(times)):
                    if times[k] == prev + 1:
                        prev = times[k]
                    else:
                        compressed_ts.append(run_start)
                        if prev != run_start:
                            compressed_ts.append(prev)
                        run_start = times[k]
                        prev = times[k]
                compressed_ts.append(run_start)
                if prev != run_start:
                    compressed_ts.append(prev)

            L = len(compressed_ts)
            num_pairs = L * (L - 1) // 2 if L >= 2 else 0
            vertex_stats.append({
                'vertex': vertex,
                'visit_count': len(times),
                'compressed_count': L,
                'actions': num_pairs,
                'times': times[:10],  # First 10 times
            })

        vertex_stats.sort(key=lambda x: x['actions'], reverse=True)
        print(f"\nTop 5 vertices by action count:")
        for stat in vertex_stats[:5]:
            print(f"  Vertex {stat['vertex']}: {stat['actions']} actions")
            print(f"    Visit count: {stat['visit_count']}, Compressed: {stat['compressed_count']}")
            print(f"    First 10 times: {stat['times']}")


def create_simple_test_trajectory():
    """Create a simple test trajectory to verify the analysis."""
    # 2 agents, 10 timesteps
    # Agent 0: A -> B -> A -> B -> A (oscillating)
    # Agent 1: C -> C -> C -> C -> C (stationary)
    M = [
        ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A'],  # Agent 0: oscillates
        ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],  # Agent 1: stationary
    ]
    return M


def main():
    # First test with simple trajectory
    print("=" * 60)
    print("TEST 1: Simple oscillating trajectory")
    print("=" * 60)
    M_simple = create_simple_test_trajectory()
    analyze_trajectory(M_simple)

    # Test with worst case: agent oscillates between 2 vertices for T steps
    print("\n" + "=" * 60)
    print("TEST 2: Worst case - oscillating agent")
    print("=" * 60)
    T = 128
    # Agent oscillates A-B-A-B-...
    M_worst = [['A' if k % 2 == 0 else 'B' for k in range(T + 1)]]
    analyze_trajectory(M_worst)

    # Calculate theoretical worst case for N=64, T=128
    print("\n" + "=" * 60)
    print("TEST 3: Theoretical analysis for N=64, T=128")
    print("=" * 60)
    N = 64
    T = 128

    # If agent visits vertex X at times t1, t2, ..., tk (non-consecutive)
    # Number of actions = k*(k-1)/2

    # Worst case: agent oscillates between 2 vertices
    # Visits each vertex ~T/2 times
    visits_per_vertex = (T + 1) // 2  # ~64-65 visits
    actions_per_vertex = visits_per_vertex * (visits_per_vertex - 1) // 2
    worst_case_per_agent = 2 * actions_per_vertex
    worst_case_total = N * worst_case_per_agent

    print(f"If each agent oscillates between 2 vertices:")
    print(f"  Visits per vertex: ~{visits_per_vertex}")
    print(f"  Actions per vertex: {actions_per_vertex}")
    print(f"  Actions per agent: {worst_case_per_agent}")
    print(f"  Total actions (N={N}): {worst_case_total}")

    # More realistic: agent visits ~20 unique vertices, each ~6 times on average
    print(f"\nMore realistic scenario:")
    avg_unique_vertices = 20
    avg_visits_per_vertex = (T + 1) / avg_unique_vertices
    # But visits are often consecutive (agent stays at vertex)
    # Assume 50% of visits are non-consecutive
    effective_visits = avg_visits_per_vertex * 0.5
    actions_per_vertex_realistic = effective_visits * (effective_visits - 1) / 2
    realistic_per_agent = avg_unique_vertices * actions_per_vertex_realistic
    realistic_total = N * realistic_per_agent

    print(f"  Unique vertices per agent: ~{avg_unique_vertices}")
    print(f"  Avg visits per vertex: ~{avg_visits_per_vertex:.1f}")
    print(f"  Effective non-consecutive visits: ~{effective_visits:.1f}")
    print(f"  Actions per agent: ~{realistic_per_agent:.0f}")
    print(f"  Total actions (N={N}): ~{realistic_total:.0f}")


if __name__ == "__main__":
    main()
