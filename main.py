"""
CyberDojo — Main CLI Entry Point

Usage:
    python main.py train        — Start co-evolutionary training
    python main.py battle       — Run a single battle with visualization
    python main.py benchmark    — Run agents against scripted baselines
    python main.py dashboard    — Start the visualization dashboard
"""

import argparse
import sys
import logging

from cyberdojo.config import CyberDojoConfig
from cyberdojo.trainer import CoEvolutionaryTrainer
from cyberdojo.agents import (
    RedTeamAgent, BlueTeamAgent,
    ScriptedRedAgent, ScriptedBlueAgent, RandomAgent,
)
from cyberdojo.llm_agents import LLMRedAgent, LLMBlueAgent, CyberCommanderAgent, CyberRedCommanderAgent, HAS_LANGCHAIN
from cyberdojo.apt_swarm import APTSwarmAgent
from cyberdojo.llm_scenario import generate_scenario
from cyberdojo.environment import CyberDojoEnv, RedAction, BlueAction


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args) -> None:
    """Run co-evolutionary training."""
    config = CyberDojoConfig()

    # Apply CLI overrides
    if args.episodes:
        config.training.total_episodes = args.episodes
    if args.steps:
        config.training.steps_per_episode = args.steps
    if args.network_size:
        sizes = {"small": config.network.small, "medium": config.network.medium,
                 "large": config.network.large}
        config.network = sizes.get(args.network_size, config.network.small)()
    if args.lr:
        config.training.learning_rate = args.lr

    trainer = CoEvolutionaryTrainer(config=config, verbose=1)

    # Optionally connect to dashboard
    if args.visualize:
        try:
            from dashboard.server import DashboardBridge
            bridge = DashboardBridge(config.dashboard)
            trainer.set_dashboard_callback(bridge.push_update)
            bridge.start_async()
            print(f"\n  🖥️  Dashboard: http://{config.dashboard.host}:{config.dashboard.port}\n")
        except ImportError:
            print("  ⚠️  Dashboard dependencies not available. Training without visualization.\n")

    results = trainer.train(total_episodes=config.training.total_episodes)

    # Save final agents
    trainer.red_agent.save("./checkpoints/red_final")
    trainer.blue_agent.save("./checkpoints/blue_final")
    print("\n  ✅ Agents saved to ./checkpoints/")


def cmd_battle(args) -> None:
    """Run a single battle."""
    config = CyberDojoConfig()
    if args.network_size:
        sizes = {"small": config.network.small, "medium": config.network.medium,
                 "large": config.network.large}
        config.network = sizes.get(args.network_size, config.network.small)()

    if args.scenario:
        print(f"\n  🧠 Generating Infinite Scenario: '{args.scenario}'...")
        scenario = generate_scenario(args.scenario)
        if scenario:
            config.network.scenario_data = scenario
            print(f"  ✨ Generated scenario with {len(scenario['subnets'])} subnets and {len(scenario['nodes'])} nodes.")
        else:
            print("  ⚠️ Failed to generate scenario. Defaulting to random.")
        
    if config.network.scenario_data:
        n_nodes = len(config.network.scenario_data["nodes"])
    else:
        n_nodes = sum(config.network.nodes_per_subnet)

    # Set up agents
    if args.red == "scripted":
        red = ScriptedRedAgent(n_nodes)
    elif args.red == "random":
        red = RandomAgent("red", RedAction.NUM_ACTIONS, n_nodes)
    elif args.red == "llm":
        red = LLMRedAgent(n_nodes)
    elif args.red == "commander":
        red = CyberRedCommanderAgent(n_nodes)
    elif args.red == "swarm":
        red = APTSwarmAgent(n_nodes)
    else:
        red = RedTeamAgent()
        if args.red_checkpoint:
            red.load(args.red_checkpoint)

    if args.blue == "scripted":
        blue = ScriptedBlueAgent(n_nodes)
    elif args.blue == "random":
        blue = RandomAgent("blue", BlueAction.NUM_ACTIONS, n_nodes)
    elif args.blue == "llm":
        blue = LLMBlueAgent(n_nodes)
    elif args.blue == "commander":
        blue = CyberCommanderAgent(n_nodes)
    else:
        blue = BlueTeamAgent()
        if args.blue_checkpoint:
            blue.load(args.blue_checkpoint)

    trainer = CoEvolutionaryTrainer(
        config=config, red_agent=red, blue_agent=blue, verbose=1
    )

    # Connect to dashboard if requested
    if args.visualize:
        try:
            from dashboard.server import DashboardBridge
            bridge = DashboardBridge(config.dashboard)
            # Tell the dashboard which team the human commands
            if args.red == "commander":
                bridge.set_commander_mode("red")
            elif args.blue == "commander":
                bridge.set_commander_mode("blue")
            trainer.set_dashboard_callback(bridge.push_update)
            bridge.start_async()
            print(f"\n  🖥️  Dashboard: http://{config.dashboard.host}:{config.dashboard.port}\n")
        except ImportError:
            pass

    result = trainer.run_single_battle(visualize=not args.visualize)

    print(f"\n  {'🔴 RED WINS!' if result.winner == 'red' else '🔵 BLUE WINS!' if result.winner == 'blue' else '🤝 DRAW'}")
    print(f"  Score — Red: {result.red_score:.1f} | Blue: {result.blue_score:.1f}")
    print(f"  Compromised: {result.compromised_nodes}/{result.total_nodes}")
    print(f"  Data Stolen: {result.data_stolen:.1f}")
    print(f"  Detections: {result.detections}")
    print(f"  Steps: {result.steps}\n")

    if getattr(args, "sim2real", False):
        try:
            from cyberdojo.sim2real import export_campaign
            export_campaign(result.events)
        except ImportError as e:
            print(f"  [!] Failed to export sim2real campaign: {e}")

    if args.visualize:
        import time
        print("  Dashboard will remain open for 15 seconds so you can view the final state...")
        time.sleep(15)


def cmd_benchmark(args) -> None:
    """Benchmark trained agents against baselines."""
    config = CyberDojoConfig()
    trainer = CoEvolutionaryTrainer(config=config, verbose=1)

    # Load trained agents if checkpoints provided
    if args.red_checkpoint:
        trainer.red_agent.load(args.red_checkpoint)
    if args.blue_checkpoint:
        trainer.blue_agent.load(args.blue_checkpoint)

    print("\n  📊 Running benchmarks...\n")
    results = trainer.benchmark()

    for matchup, data in results.items():
        print(f"  {matchup}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
        print()


def cmd_demo(args) -> None:
    """Run continuous demo battles with dashboard visualization."""
    import time
    import threading

    config = CyberDojoConfig()
    if args.network_size:
        sizes = {"small": config.network.small, "medium": config.network.medium,
                 "large": config.network.large}
        config.network = sizes.get(args.network_size, config.network.small)()

    n_nodes = sum(config.network.nodes_per_subnet)

    # Start the dashboard server in background
    try:
        from dashboard.server import DashboardBridge, start_dashboard
        bridge = DashboardBridge(config.dashboard)
        bridge.start_async()
    except ImportError as e:
        print(f"  ❌ Error: {e}")
        print("  Run: pip install flask flask-socketio")
        return

    port = args.port or config.dashboard.port
    print(f"\n  ⚔️  CYBERDOJO — Live Demo Mode")
    print(f"  🖥️  Dashboard: http://{config.dashboard.host}:{port}")
    print(f"  📡 Open the link above in your browser!")
    print(f"  ⏹️  Press Ctrl+C to stop\n")

    # Give server a moment to start
    time.sleep(1.0)

    red_elo = 1000.0
    blue_elo = 1000.0
    battle_num = 0
    try:
        while True:
            battle_num += 1

            import os
            # Alternate agent types for variety
            if battle_num % 4 == 0:
                red = RandomAgent("red", RedAction.NUM_ACTIONS, n_nodes)
                blue = ScriptedBlueAgent(n_nodes)
                matchup = "Random Red vs Scripted Blue"
            elif battle_num % 4 == 1:
                red = ScriptedRedAgent(n_nodes)
                blue = ScriptedBlueAgent(n_nodes)
                matchup = "Scripted Red vs Scripted Blue"
            elif battle_num % 4 == 2:
                red = ScriptedRedAgent(n_nodes)
                blue = RandomAgent("blue", BlueAction.NUM_ACTIONS, n_nodes)
                matchup = "Scripted Red vs Random Blue"
            else:
                if HAS_LANGCHAIN and os.environ.get("OPENAI_API_KEY"):
                    red = LLMRedAgent(n_nodes)
                    blue = ScriptedBlueAgent(n_nodes)
                    matchup = "🤖 LLM Red vs Scripted Blue"
                else:
                    red = ScriptedRedAgent(n_nodes)
                    blue = ScriptedBlueAgent(n_nodes)
                    matchup = "Scripted Red vs Scripted Blue"

            print(f"  ⚔️  Battle #{battle_num}: {matchup}")

            # Create environment
            env = CyberDojoEnv(config=config, mode="red")
            env.set_opponent_policy(lambda obs, b=blue: b.act(obs))

            obs, info = env.reset()

            # Push initial network topology
            bridge.push_update({
                "step": 0,
                "network_state": env.network.get_topology_data(),
                "red": {"action": "preparing", "target": None, "events": {}},
                "blue": {"action": "preparing", "target": None, "events": {}},
            })
            time.sleep(1.5)

            for step in range(config.training.steps_per_episode):
                action = red.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                # Push the step data directly — it already has red/blue actions,
                # targets, events, and full network state from _log_event()
                bridge.push_update(env.step_data)

                time.sleep(0.8)  # Pace for visual observation

                if terminated or truncated:
                    break

            # Determine winner
            compromised_ratio = info.get("compromised_ratio", 0)
            data_stolen_ratio = info.get("data_stolen_ratio", 0)
            red_score = compromised_ratio * 50 + data_stolen_ratio * 50
            blue_score = (1 - compromised_ratio) * 30 + (1 - data_stolen_ratio) * 30

            if red_score > blue_score + 10:
                winner = "🔴 RED WINS"
                winner_key = "red"
            elif blue_score > red_score + 10:
                winner = "🔵 BLUE WINS"
                winner_key = "blue"
            else:
                winner = "🤝 DRAW"
                winner_key = "draw"

            # Update Elo-style ratings
            e_red = 1.0 / (1.0 + 10 ** ((blue_elo - red_elo) / 400.0))
            if winner_key == "red":
                red_elo += 32 * (1.0 - e_red)
                blue_elo += 32 * (0.0 - (1.0 - e_red))
            elif winner_key == "blue":
                red_elo += 32 * (0.0 - e_red)
                blue_elo += 32 * (1.0 - (1.0 - e_red))
            else:
                red_elo += 32 * (0.5 - e_red)
                blue_elo += 32 * (0.5 - (1.0 - e_red))

            # Push training progress to dashboard charts
            bridge.push_training_progress({
                "red_elo": red_elo,
                "blue_elo": blue_elo,
                "battle": {
                    "episode": battle_num,
                    "red_score": red_score,
                    "blue_score": blue_score,
                    "winner": winner_key,
                },
            })

            comp_nodes = info.get("compromised_nodes", 0)
            total_nodes = info.get("total_nodes", n_nodes)
            print(f"        {winner} | Compromised: {comp_nodes}/{total_nodes} | "
                  f"Red: {red_score:.0f} Blue: {blue_score:.0f} | "
                  f"Elo R:{red_elo:.0f} B:{blue_elo:.0f}")

            time.sleep(3.0)  # Pause between battles

    except KeyboardInterrupt:
        print("\n\n  ⏹️  Demo stopped. Thanks for watching!")


def cmd_dashboard(args) -> None:
    """Start the visualization dashboard."""
    config = CyberDojoConfig()
    if args.port:
        config.dashboard.port = args.port

    try:
        from dashboard.server import start_dashboard
        print(f"\n  🖥️  Starting CyberDojo Dashboard...")
        print(f"  📡 Open http://{config.dashboard.host}:{config.dashboard.port}\n")
        start_dashboard(config.dashboard)
    except ImportError as e:
        print(f"  ❌ Error: {e}")
        print("  Run: pip install flask flask-socketio eventlet")


def main():
    parser = argparse.ArgumentParser(
        description="⚔️ CyberDojo — Adversarial AI War Games Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                               — Live demo with dashboard
  python main.py train --episodes 500               — Co-evolutionary RL training
  python main.py battle --red scripted --blue scripted
  python main.py dashboard --port 5000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command (recommended starting point)
    demo_parser = subparsers.add_parser("demo", help="Run live demo with dashboard visualization")
    demo_parser.add_argument("--network-size", choices=["small", "medium", "large"],
                              default="small")
    demo_parser.add_argument("--port", type=int, default=5000)
    demo_parser.set_defaults(func=cmd_demo)

    # Train command
    train_parser = subparsers.add_parser("train", help="Start co-evolutionary training")
    train_parser.add_argument("--episodes", type=int, help="Number of training episodes")
    train_parser.add_argument("--steps", type=int, help="Steps per episode")
    train_parser.add_argument("--network-size", choices=["small", "medium", "large"],
                              default="small")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--visualize", action="store_true",
                              help="Enable dashboard visualization")
    train_parser.set_defaults(func=cmd_train)

    # Battle command
    battle_parser = subparsers.add_parser("battle", help="Run a single battle")
    battle_parser.add_argument("--red", choices=["rl", "scripted", "random", "llm", "commander", "swarm"],
                                default="scripted")
    battle_parser.add_argument("--blue", choices=["rl", "scripted", "random", "llm", "commander"],
                                default="scripted")
    battle_parser.add_argument("--red-checkpoint", type=str)
    battle_parser.add_argument("--blue-checkpoint", type=str)
    battle_parser.add_argument("--network-size", choices=["small", "medium", "large"],
                                default="small")
    battle_parser.add_argument("--scenario", type=str,
                                help="Generate a dynamic LLM scenario (e.g., 'Hospital', 'Bank')")
    battle_parser.add_argument("--sim2real", action="store_true",
                                help="Export battle events as real-world CLI bash playbooks")
    battle_parser.add_argument("--visualize", action="store_true",
                                help="Enable dashboard visualization")
    battle_parser.set_defaults(func=cmd_battle)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark against baselines")
    bench_parser.add_argument("--red-checkpoint", type=str)
    bench_parser.add_argument("--blue-checkpoint", type=str)
    bench_parser.set_defaults(func=cmd_benchmark)

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start visualization dashboard")
    dash_parser.add_argument("--port", type=int, default=5000)
    dash_parser.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(verbose=getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
