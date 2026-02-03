#!/usr/bin/env python3
"""
EarnHFT Training Script.

Trains BetaAgents with different risk preferences for the EarnHFT system.

Usage:
    # Train a single agent with specific beta
    python train_earnhft.py --beta 0.5 --epochs 100
    
    # Train all default betas (0.1, 0.3, 0.5, 0.7, 0.9)
    python train_earnhft.py --all-betas --epochs 100
    
    # Build pool from trained agents
    python train_earnhft.py --build-pool
"""
import argparse
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.earnhft import BetaAgent, AgentPool


def train_single_agent(beta: float, epochs: int = 100, save_dir: str = "earnhft_agents"):
    """Train a single BetaAgent."""
    print(f"\n{'='*60}")
    print(f"Training BetaAgent with β={beta:.1f}")
    print(f"{'='*60}")
    
    agent = BetaAgent(beta=beta)
    agent.training = True
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # The actual training happens in the main trading loop
    # Here we just set up the agent and save initial state
    agent.save(os.path.join(save_dir, f"agent_beta_{beta:.1f}"))
    
    print(f"[Train] Agent β={beta:.1f} initialized and saved to {save_dir}")
    print(f"[Train] To train: run `python run.py earnhft --train --beta {beta}`")
    
    return agent


def train_all_betas(betas: list = None, epochs: int = 100, save_dir: str = "earnhft_agents"):
    """Train agents with all specified betas."""
    if betas is None:
        betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\n{'='*60}")
    print(f"Training {len(betas)} BetaAgents: {betas}")
    print(f"{'='*60}")
    
    for beta in betas:
        train_single_agent(beta, epochs, save_dir)
    
    print(f"\n[Train] All agents initialized in {save_dir}")


def build_pool(save_dir: str = "earnhft_agents"):
    """Build agent pool from trained agents."""
    print(f"\n{'='*60}")
    print(f"Building Agent Pool")
    print(f"{'='*60}")
    
    pool = AgentPool(save_dir)
    trained = pool.list_trained_agents()
    
    if not trained:
        print("[Pool] No trained agents found. Train agents first.")
        return
    
    print(f"[Pool] Found {len(trained)} trained agents:")
    for beta, path in trained:
        print(f"  β={beta:.1f}: {path}")
    
    # Build pool with default regime mapping
    pool.build_pool_from_trained()
    
    # Save config
    pool.save_pool_config()
    
    print(f"\n[Pool] Pool built with {len(pool)} regime mappings")


def main():
    parser = argparse.ArgumentParser(description="EarnHFT Training Script")
    parser.add_argument("--beta", type=float, help="Beta value for single agent training")
    parser.add_argument("--all-betas", action="store_true", help="Train all default betas")
    parser.add_argument("--betas", type=str, help="Comma-separated beta values, e.g. '0.1,0.3,0.5'")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--save-dir", type=str, default="earnhft_agents", help="Directory to save agents")
    parser.add_argument("--build-pool", action="store_true", help="Build pool from trained agents")
    
    args = parser.parse_args()
    
    if args.build_pool:
        build_pool(args.save_dir)
    elif args.all_betas:
        train_all_betas(epochs=args.epochs, save_dir=args.save_dir)
    elif args.betas:
        betas = [float(b.strip()) for b in args.betas.split(",")]
        train_all_betas(betas, args.epochs, args.save_dir)
    elif args.beta is not None:
        train_single_agent(args.beta, args.epochs, args.save_dir)
    else:
        print("Usage:")
        print("  python train_earnhft.py --beta 0.5         # Train single agent")
        print("  python train_earnhft.py --all-betas        # Train all defaults")
        print("  python train_earnhft.py --build-pool       # Build pool")


if __name__ == "__main__":
    main()
